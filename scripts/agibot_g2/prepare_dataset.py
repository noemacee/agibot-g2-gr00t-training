# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26,<3", "pyarrow>=17,<24", "scipy>=1.12,<2"]
# ///
"""Prepare G2 LeRobot v2/v3 snapshots for GR00T without modifying the source."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


BODY_NAMES = (
    [f"idx0{i}_body_joint{i}" for i in range(1, 6)]
    + [f"idx1{i}_head_joint{i}" for i in range(1, 4)]
    + [f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)]
    + [f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)]
)
HAND_SUFFIXES = [
    (1, "thumb_roll_joint"),
    (2, "thumb_abad_joint"),
    (3, "thumb_mcp_joint"),
    (6, "index_abad_joint"),
    (7, "index_pip_joint"),
    (9, "middle_pip_joint"),
    (11, "ring_abad_joint"),
    (12, "ring_pip_joint"),
    (14, "pinky_abad_joint"),
    (15, "pinky_pip_joint"),
]
HAND_NAMES = [
    f"idx{base + index}_hand_{side}_{suffix}"
    for base, side in [(30, "l"), (70, "r")]
    for index, suffix in HAND_SUFFIXES
]
JOINT_NAMES = BODY_NAMES + HAND_NAMES
CAMERAS = {
    "head": "observation.images.head_stereo_left",
    "left_wrist": "observation.images.hand_left_color",
    "right_wrist": "observation.images.hand_right_color",
}
LAYOUTS = {
    "joints": {
        "body": (0, 5),
        "head": (5, 8),
        "left_arm": (8, 15),
        "right_arm": (15, 22),
        "left_hand": (22, 32),
        "right_hand": (32, 42),
    },
    "eef": {
        "body": (0, 5),
        "head": (5, 8),
        "left_eef": (8, 14),
        "right_eef": (14, 20),
        "left_hand": (20, 30),
        "right_hand": (30, 40),
    },
}
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
ANNOTATION = "annotation.human.task_description"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}")
    return int(value)


class Source:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.info = read_json(self.root / "meta/info.json")
        version = self.info["codebase_version"]
        if version not in ("v2.0", "v2.1", "v3.0"):
            raise ValueError(f"Unsupported LeRobot version: {version}")
        self.v3 = version == "v3.0"
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.used_files = {self.root / "meta/info.json"}
        if self.v3:
            paths = sorted((self.root / "meta/episodes").rglob("*.parquet"))
            self.episodes = [row for p in paths for row in pq.read_table(p).to_pylist()]
            task_path = self.root / "meta/tasks.parquet"
            table = pq.read_table(task_path)
            # LeRobot v3 stores task text as the pandas index in tasks.parquet.
            pandas_meta = json.loads((table.schema.metadata or {}).get(b"pandas", b"{}"))
            index_columns = pandas_meta.get("index_columns", [])
            task_key = (
                "task"
                if "task" in table.column_names
                else next(
                    (k for k in index_columns if isinstance(k, str) and k in table.column_names),
                    None,
                )
            )
            if task_key is None:
                raise ValueError("Cannot identify task text in meta/tasks.parquet")
            tasks = [
                {"task_index": r["task_index"], "task": r[task_key]} for r in table.to_pylist()
            ]
            self.used_files.update(paths + [task_path])
        else:

            def lines(name):
                p = self.root / "meta" / name
                self.used_files.add(p)
                return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

            self.episodes = lines("episodes.jsonl")
            tasks = lines("tasks.jsonl")
        self.tasks = {}
        for task in tasks:
            key = integer(task["task_index"], "task_index")
            if key in self.tasks or not isinstance(task["task"], str) or not task["task"].strip():
                raise ValueError("Task IDs must be unique and task text nonempty")
            self.tasks[key] = task["task"]
        ids = [integer(e["episode_index"], "episode_index") for e in self.episodes]
        if len(set(ids)) != len(ids) or len(ids) != self.info["total_episodes"]:
            raise ValueError("Duplicate episode IDs or episode count inconsistent with info.json")
        self.episodes.sort(key=lambda e: e["episode_index"])
        if (
            sum(integer(e["length"], "episode length") for e in self.episodes)
            != self.info["total_frames"]
        ):
            raise ValueError("Episode lengths do not sum to total_frames")
        self.orders = {}
        for key, names in [
            ("observation.state", BODY_NAMES),
            ("action", BODY_NAMES),
            ("observation.hand_position", HAND_NAMES),
            ("hand_target", HAND_NAMES),
            ("hand_target_valid", HAND_NAMES),
        ]:
            feature = self.info["features"][key]
            actual = feature.get("names")
            if (
                not isinstance(actual, list)
                or len(actual) != len(names)
                or set(actual) != set(names)
            ):
                raise ValueError(
                    f"{key}: expected the named G2 joints exactly; inspect schema changes"
                )
            if feature["shape"] != [len(names)]:
                raise ValueError(f"{key}: unexpected shape")
            self.orders[key] = [actual.index(name) for name in names]
        for key in CAMERAS.values():
            feature = self.info["features"][key]
            if feature["dtype"] != "video":
                raise ValueError(f"{key} must be a video feature")
            video_info = feature.get("info", feature.get("video_info", {}))
            if not np.isclose(video_info.get("video.fps", self.fps), self.fps):
                raise ValueError(f"{key}: videos must already use the dataset frame rate")
        paths = sorted((self.root / "data").rglob("*.parquet"))
        if not paths:
            raise ValueError("No source data parquet files")
        self.used_files.update(paths)
        self.data = ds.dataset(paths, format="parquet")
        if self.data.count_rows() != self.info["total_frames"]:
            raise ValueError("Parquet row count disagrees with total_frames")

    def episode(self, record):
        eid, length = record["episode_index"], record["length"]
        columns = list(self.orders) + ["timestamp", "frame_index", "index", "task_index"]
        table = self.data.to_table(filter=ds.field("episode_index") == eid, columns=columns)
        table = table.sort_by("frame_index")
        if len(table) != length or not np.array_equal(
            table["frame_index"].to_numpy(), np.arange(length)
        ):
            raise ValueError(f"Episode {eid}: missing/duplicate frames or incorrect length")
        time = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        if not np.all(np.isfinite(time)) or not np.allclose(
            time, np.arange(length) / self.fps, atol=1e-4, rtol=0
        ):
            raise ValueError(f"Episode {eid}: timestamps must be a zero-based regular fps grid")
        tasks = np.asarray(table["task_index"].to_pylist())
        if tasks.dtype.kind not in "iu" or not set(tasks.tolist()) <= self.tasks.keys():
            raise ValueError(f"Episode {eid}: invalid task IDs")
        indices = table["index"].to_numpy()
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(np.diff(indices) != 1):
            raise ValueError(f"Episode {eid}: invalid global frame indices")
        if self.v3 and not np.array_equal(
            indices, np.arange(record["dataset_from_index"], record["dataset_to_index"])
        ):
            raise ValueError(f"Episode {eid}: global frame indices disagree with episode metadata")
        values = {}
        for key, order in self.orders.items():
            arr = np.asarray(table[key].to_pylist(), dtype=np.float64)
            if arr.shape != (length, len(order)):
                raise ValueError(f"Episode {eid}: invalid {key} shape")
            values[key] = arr[:, order]
        valid = values["hand_target_valid"]
        if not np.all(np.isin(valid, [0, 1])):
            raise ValueError(f"Episode {eid}: hand validity must contain only 0/1")
        for key in ("observation.state", "action", "observation.hand_position"):
            if not np.isfinite(values[key]).all():
                raise ValueError(f"Episode {eid}: nonfinite {key}")
        if not np.isfinite(values["hand_target"][valid == 1]).all():
            raise ValueError(f"Episode {eid}: nonfinite valid hand target")
        state = np.concatenate(
            [values["observation.state"], values["observation.hand_position"]], axis=1
        )
        action = np.concatenate([values["action"], values["hand_target"]], axis=1)
        return state, action, np.all(valid == 1, axis=1), tasks, indices

    def video(self, record, key):
        eid = record["episode_index"]
        if self.v3:
            prefix = "videos/" + key
            path = self.root / self.info["video_path"].format(
                video_key=key,
                chunk_index=record[prefix + "/chunk_index"],
                file_index=record[prefix + "/file_index"],
            )
            start, end = record[prefix + "/from_timestamp"], record[prefix + "/to_timestamp"]
            if (
                not np.isfinite([start, end]).all()
                or start < 0
                or not np.isclose(end - start, record["length"] / self.fps, atol=1e-4, rtol=0)
            ):
                raise ValueError(f"Episode {eid}: inconsistent {key} video timestamps")
        else:
            path = self.root / self.info["video_path"].format(
                video_key=key,
                episode_chunk=eid // self.info["chunks_size"],
                episode_index=eid,
            )
            start = 0.0
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Missing video or path outside source: {path}")
        self.used_files.add(path)
        # These exports are CFR, with one encoded video frame per dataset row.
        first_frame = round(start * self.fps)
        if abs(first_frame / self.fps - start) > 1e-4:
            raise ValueError("Video episode offset is not on the frame grid")
        return path, first_frame


def split_assignments(manifest, episodes):
    allowed = {"train", "validation", "test", "exclude", "recording_groups"}
    if set(manifest) - allowed:
        raise ValueError("Unknown split manifest keys")
    assignment = {}
    for split in ("train", "validation", "test", "exclude"):
        for eid in manifest.get(split, []):
            eid = integer(eid, "split episode ID")
            if eid in assignment:
                raise ValueError(f"Episode {eid} assigned more than once")
            assignment[eid] = split
    if set(assignment) != {e["episode_index"] for e in episodes}:
        raise ValueError(
            "Split manifest must assign every source episode exactly once (or exclude it)"
        )
    if "train" not in assignment.values():
        raise ValueError("At least one training episode is required")
    groups = {}
    recording_groups = manifest.get("recording_groups", {})
    if set(recording_groups) - {str(eid) for eid in assignment}:
        raise ValueError("recording_groups references an unknown episode")
    for eid, split in assignment.items():
        if split == "exclude":
            continue
        group = str(recording_groups.get(str(eid), f"episode:{eid}"))
        if group in groups and groups[group] != split:
            raise ValueError(f"Original recording {group} leaks across splits")
        groups[group] = split
    return assignment


def calibration_arrays(calibration):
    entries = calibration["joints"]
    if set(entries) != set(JOINT_NAMES):
        raise ValueError("Calibration must describe all 42 named joints exactly")
    result = {}
    for signal in ("state", "action"):
        scale, offset = [], []
        for name in JOINT_NAMES:
            entry = entries[name]
            if entry["unit"] not in ("rad", "m"):
                raise ValueError(f"{name}: output unit must explicitly be rad or m")
            s, o = entry[signal]["scale"], entry[signal]["offset"]
            if s is None or o is None or not np.isfinite([s, o]).all() or s == 0:
                raise ValueError(
                    f"{name}: provide finite nonzero scale and finite offset for {signal}"
                )
            scale.append(s)
            offset.append(o)
        result[signal] = (np.asarray(scale), np.asarray(offset))
    return result, [entries[name]["unit"] for name in JOINT_NAMES]


def load_fk(path):
    spec = importlib.util.spec_from_file_location("g2_fk_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = module.KINEMATICS
    for key in ("frame", "left_tcp", "right_tcp", "model_sha256"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"FK provider KINEMATICS must document {key}")
    if not callable(module.compute_poses):
        raise ValueError("FK provider must define compute_poses(joints, joint_names, units)")
    return module


def eef_values(joints, provider, units):
    # Pass a copy so a provider cannot change the companion joint training view.
    poses = np.asarray(
        provider.compute_poses(joints.copy(), list(JOINT_NAMES), list(units)), dtype=np.float64
    )
    if poses.shape != (len(joints), 2, 7) or not np.isfinite(poses).all():
        raise ValueError("FK must return finite (N, 2, 7) XYZ metres + XYZW quaternion poses")
    quat = poses[:, :, 3:]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.allclose(norm, 1, atol=1e-3, rtol=0):
        raise ValueError("FK must return unit quaternions (tolerance 1e-3)")
    rotvec = Rotation.from_quat((quat / norm).reshape(-1, 4)).as_rotvec().reshape(-1, 2, 3)
    arms = np.concatenate([poses[:, :, :3], rotvec], axis=2).reshape(-1, 12)
    return np.concatenate([joints[:, :8], arms, joints[:, 22:]], axis=1)


def valid_segments(valid, minimum):
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= minimum]


def probe_video(path, ffprobe):
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height,avg_frame_rate,start_time:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    decoded = json.loads(result.stdout)
    stream = decoded["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    times = np.array([float(f["best_effort_timestamp_time"]) for f in decoded["frames"]])
    if len(times) != int(stream["nb_read_frames"]) or not np.allclose(
        times, np.arange(len(times)) / fps, atol=1e-4, rtol=0
    ):
        raise ValueError(f"Video must have a zero-based constant frame-rate timeline: {path}")
    return {
        "frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "start": float(stream.get("start_time", 0)),
    }


def cut_video(src, dst, first, length, fps, ffmpeg, ffprobe):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Decode and trim by frame number: stream copy can seek to the wrong keyframe.
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-vf",
            f"trim=start_frame={first}:end_frame={first + length},setpts=N/({fps}*TB)",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-fps_mode",
            "cfr",
            "-n",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    actual = probe_video(dst, ffprobe)
    if (
        actual["frames"] != length
        or not np.isclose(actual["fps"], fps)
        or abs(actual["start"]) > 1e-4
    ):
        raise ValueError(f"Output video does not match the requested frame grid: {dst}")
    return actual


def modality(variant):
    groups = {k: {"start": a, "end": b} for k, (a, b) in LAYOUTS[variant].items()}
    return {
        "state": groups,
        "action": deepcopy(groups),
        "video": {k: {"original_key": v} for k, v in CAMERAS.items()},
        "annotation": {"human.task_description": {"original_key": ANNOTATION}},
    }


def feature(dtype, names):
    return {"dtype": dtype, "shape": [len(names)], "names": names}


def prepare(
    source,
    output,
    calibration,
    splits,
    variants,
    revision,
    provider=None,
    minimum=16,
    ffmpeg="ffmpeg",
    ffprobe="ffprobe",
):
    if not variants or set(variants) - {"joints", "eef"} or len(set(variants)) != len(variants):
        raise ValueError("Choose unique variants from joints and eef")
    if minimum < 16:
        raise ValueError("minimum segment length must be >= the templates' 16-step horizon")
    if not revision.strip():
        raise ValueError("Record an immutable source revision or local snapshot identifier")
    if "eef" in variants and provider is None:
        raise ValueError(
            "EEF preparation requires a calibrated FK provider; observed poses are not command labels"
        )
    scales, units = calibration_arrays(calibration)
    assignments = split_assignments(splits, source.episodes)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source.root) or source.root.is_relative_to(output):
        raise ValueError("Output must be new and outside the source dataset")
    if not shutil.which(ffmpeg) or not shutil.which(ffprobe):
        raise ValueError("ffmpeg and ffprobe are required (or supply their executable paths)")
    for record in source.episodes:
        if assignments[record["episode_index"]] != "exclude":
            for key in CAMERAS.values():
                source.video(record, key)
    source_hashes = {str(p.relative_to(source.root)): sha256(p) for p in sorted(source.used_files)}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        manifest = {
            "source_revision": revision,
            "source_info_sha256": sha256(source.root / "meta/info.json"),
            "adapter_sha256": sha256(Path(__file__)),
            "splits": splits,
            "calibration": calibration,
            "minimum_segment_frames": minimum,
            "variants": variants,
            "episodes": [],
            "statistics": "Not generated; run GR00T stats on train, reuse for validation/test.",
        }
        if provider is not None:
            manifest["kinematics"] = provider.KINEMATICS
            manifest["fk_provider_sha256"] = sha256(Path(provider.__file__))
        records = {s: [] for s in ("train", "validation", "test")}
        counts = dict.fromkeys(records, 0)
        video_features = {k: deepcopy(source.info["features"][k]) for k in CAMERAS.values()}
        probes = {}
        for record in source.episodes:
            eid = record["episode_index"]
            split = assignments[eid]
            if split == "exclude":
                manifest["episodes"].append({"source_episode": eid, "split": split})
                continue
            state, action, valid, tasks, indices = source.episode(record)
            segments = valid_segments(valid, minimum)
            audit = {
                "source_episode": eid,
                "split": split,
                "source_frames": len(valid),
                "invalid_hand_frames": int((~valid).sum()),
                "dropped_short_valid_frames": int(valid.sum() - sum(b - a for a, b in segments)),
                "segments": [],
            }
            manifest["episodes"].append(audit)
            for begin, end in segments:
                length, new_id = end - begin, len(records[split])
                values = {}
                q_state = state[begin:end] * scales["state"][0] + scales["state"][1]
                q_action = action[begin:end] * scales["action"][0] + scales["action"][1]
                if "joints" in variants:
                    values["joints"] = (q_state, q_action)
                if "eef" in variants:
                    values["eef"] = (
                        eef_values(q_state, provider, units),
                        eef_values(q_action, provider, units),
                    )
                segment = {
                    "episode_index": new_id,
                    "length": length,
                    "tasks": [source.tasks[int(i)] for i in sorted(set(tasks[begin:end]))],
                }
                records[split].append(segment)
                audit["segments"].append(
                    {"output_episode": new_id, "source_start_frame": begin, "source_end_frame": end}
                )
                for variant, (obs, act) in values.items():
                    obs, act = obs.astype(np.float32), act.astype(np.float32)
                    if not np.isfinite(obs).all() or not np.isfinite(act).all():
                        raise ValueError("Calibrated state/action overflows float32")
                    root = stage / variant / split
                    path = root / DATA_PATH.format(
                        episode_chunk=new_id // 1000, episode_index=new_id
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    table = pa.table(
                        {
                            "observation.state": pa.array(
                                obs.tolist(), type=pa.list_(pa.float32(), obs.shape[1])
                            ),
                            "action": pa.array(
                                act.tolist(), type=pa.list_(pa.float32(), act.shape[1])
                            ),
                            "timestamp": pa.array(
                                np.arange(length) / source.fps, type=pa.float32()
                            ),
                            "frame_index": pa.array(np.arange(length), type=pa.int64()),
                            "episode_index": pa.array(np.full(length, new_id), type=pa.int64()),
                            "index": pa.array(
                                np.arange(counts[split], counts[split] + length), type=pa.int64()
                            ),
                            "task_index": pa.array(tasks[begin:end], type=pa.int64()),
                            ANNOTATION: pa.array(tasks[begin:end], type=pa.int64()),
                            "source_episode_index": pa.array(np.full(length, eid), type=pa.int64()),
                            "source_frame_index": pa.array(np.arange(begin, end), type=pa.int64()),
                            "source_global_index": pa.array(indices[begin:end], type=pa.int64()),
                        }
                    )
                    pq.write_table(table, path)
                for key in CAMERAS.values():
                    src, first = source.video(record, key)
                    if src not in probes:
                        probes[src] = probe_video(src, ffprobe)
                    p = probes[src]
                    if (
                        p["frames"] < first + record["length"]
                        or not np.isclose(p["fps"], source.fps)
                        or abs(p["start"]) > 1e-4
                    ):
                        raise ValueError(
                            f"Source video does not match episode frame metadata: {src}"
                        )
                    if source.info["features"][key]["shape"] != [p["height"], p["width"], 3]:
                        raise ValueError(f"Video dimensions disagree with metadata: {src}")
                    rel = VIDEO_PATH.format(
                        episode_chunk=new_id // 1000, episode_index=new_id, video_key=key
                    )
                    dst = stage / variants[0] / split / rel
                    cut_video(src, dst, first + begin, length, source.fps, ffmpeg, ffprobe)
                    for variant in variants[1:]:
                        other = stage / variant / split / rel
                        other.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(dst, other)
                    video_features[key].pop("video_info", None)
                    video_features[key]["info"] = {
                        "video.fps": source.fps,
                        "video.height": p["height"],
                        "video.width": p["width"],
                        "video.channels": 3,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    }
                counts[split] += length
        for split, episodes in records.items():
            if not episodes:
                if split in assignments.values():
                    raise ValueError(f"No usable segments remain in {split}; inspect hand validity")
                continue
            for variant in variants:
                root = stage / variant / split
                names = (
                    JOINT_NAMES
                    if variant == "joints"
                    else (
                        JOINT_NAMES[:8]
                        + [
                            f"{side}_eef.{axis}"
                            for side in ("left", "right")
                            for axis in ("x", "y", "z", "rx", "ry", "rz")
                        ]
                        + HAND_NAMES
                    )
                )
                features = deepcopy(video_features)
                features.update(
                    {
                        "observation.state": feature("float32", names),
                        "action": feature("float32", names),
                    }
                )
                for key in (
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                    ANNOTATION,
                    "source_episode_index",
                    "source_frame_index",
                    "source_global_index",
                ):
                    features[key] = feature("int64", [key])
                features["timestamp"] = feature("float32", ["timestamp"])
                info = {
                    "codebase_version": "v2.1",
                    "robot_type": "agibot_g2",
                    "fps": source.fps,
                    "total_episodes": len(episodes),
                    "total_frames": counts[split],
                    "total_tasks": len(source.tasks),
                    "total_videos": len(episodes) * len(CAMERAS),
                    "total_chunks": (len(episodes) + 999) // 1000,
                    "chunks_size": 1000,
                    "splits": {split: f"0:{len(episodes)}"},
                    "data_path": DATA_PATH,
                    "video_path": VIDEO_PATH,
                    "features": features,
                }
                write_json(root / "meta/info.json", info)
                write_json(root / "meta/modality.json", modality(variant))
                write_jsonl(root / "meta/episodes.jsonl", episodes)
                write_jsonl(
                    root / "meta/tasks.jsonl",
                    [{"task_index": k, "task": v} for k, v in sorted(source.tasks.items())],
                )
        manifest["output_frames"] = counts
        after_hashes = {
            str(p.relative_to(source.root)): sha256(p) for p in sorted(source.used_files)
        }
        if after_hashes != source_hashes:
            raise ValueError(
                "Source files changed during conversion; use an immutable local snapshot"
            )
        manifest["source_files"] = source_hashes
        write_json(stage / "preparation_manifest.json", manifest)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest


def inspect(source):
    report = {
        "info": {
            k: source.info[k] for k in ("codebase_version", "total_episodes", "total_frames", "fps")
        },
        "episodes": [],
        "calibration_template": {
            "joints": {
                name: {
                    "unit": None,
                    "state": {"scale": None, "offset": None},
                    "action": {"scale": None, "offset": None},
                }
                for name in JOINT_NAMES
            }
        },
        "split_template": {"train": [], "validation": [], "test": [], "exclude": []},
    }
    for record in source.episodes:
        _, _, valid, _, _ = source.episode(record)
        report["episodes"].append(
            {
                "episode_index": record["episode_index"],
                "frames": len(valid),
                "all_hand_targets_valid_frames": int(valid.sum()),
                "usable_segments_16_frames": valid_segments(valid, 16),
            }
        )
        report["split_template"]["train"].append(record["episode_index"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser(
        "inspect", help="Validate numerical schema and report hand validity; no videos needed"
    )
    audit.add_argument("--source", type=Path, required=True)
    build = sub.add_parser(
        "prepare", help="Write separate GR00T v2 datasets; never overwrite input/output"
    )
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--calibration", type=Path, required=True)
    build.add_argument("--splits", type=Path, required=True)
    build.add_argument("--source-revision", required=True)
    build.add_argument("--variants", nargs="+", choices=["joints", "eef"], default=["joints"])
    build.add_argument(
        "--fk-provider", type=Path, help="Trusted Python file implementing calibrated FK"
    )
    build.add_argument("--min-segment-frames", type=int, default=16)
    build.add_argument("--ffmpeg", default="ffmpeg")
    build.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    source = Source(args.source)
    if args.command == "inspect":
        print(json.dumps(inspect(source), indent=2))
    else:
        provider = load_fk(args.fk_provider) if args.fk_provider else None
        report = prepare(
            source,
            args.output,
            read_json(args.calibration),
            read_json(args.splits),
            args.variants,
            args.source_revision,
            provider,
            args.min_segment_frames,
            args.ffmpeg,
            args.ffprobe,
        )
        print(json.dumps({"output": str(args.output), "frames": report["output_frames"]}, indent=2))


if __name__ == "__main__":
    main()
