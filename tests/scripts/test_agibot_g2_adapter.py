# SPDX-License-Identifier: Apache-2.0
"""Synthetic data only: conversion, label validity, frame alignment, and FK contract."""

from copy import deepcopy
import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.agibot_g2 import prepare_dataset as adapter


@pytest.fixture
def calibration():
    return {
        "joints": {
            name: {
                "unit": "rad",
                "state": {"scale": 2, "offset": 0.1},
                "action": {"scale": 3, "offset": -0.2},
            }
            for name in adapter.JOINT_NAMES
        }
    }


@pytest.fixture
def binaries():
    ffmpeg = os.environ.get("G2_TEST_FFMPEG", shutil.which("ffmpeg"))
    ffprobe = os.environ.get("G2_TEST_FFPROBE", shutil.which("ffprobe"))
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe required for video integration test")
    return ffmpeg, ffprobe


def make_source(root, version="v3.0", videos=None):
    fps, lengths, ids = 20, [40, 20], [3, 9]
    features = {}
    for key, names in [
        ("observation.state", adapter.BODY_NAMES),
        ("action", adapter.BODY_NAMES),
        ("observation.hand_position", adapter.HAND_NAMES),
        ("hand_target", adapter.HAND_NAMES),
        ("hand_target_valid", adapter.HAND_NAMES),
    ]:
        # Source ordering deliberately differs from canonical/controller ordering.
        features[key] = adapter.feature("float32", list(reversed(names)))
    for key in adapter.CAMERAS.values():
        features[key] = {
            "dtype": "video",
            "shape": [16, 16, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": fps},
        }
    info = {
        "codebase_version": version,
        "total_episodes": 2,
        "total_frames": 60,
        "fps": fps,
        "chunks_size": 1000,
        "features": features,
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            if version == "v3.0"
            else adapter.VIDEO_PATH
        ),
    }
    adapter.write_json(root / "meta/info.json", info)
    rows, episodes = [], []
    for eid, length in zip(ids, lengths):
        start = len(rows)
        record = {"episode_index": eid, "length": length, "tasks": ["synthetic task"]}
        record.update({"dataset_from_index": start, "dataset_to_index": start + length})
        for key in adapter.CAMERAS.values():
            record.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": start / fps,
                    f"videos/{key}/to_timestamp": (start + length) / fps,
                }
            )
        episodes.append(record)
        for frame in range(length):
            body = np.arange(22) / 100 + frame / 1000
            hands = np.arange(20) / 100 + frame / 1000
            hand_action = hands + 0.1
            valid = np.ones(20)
            if eid == 3 and frame == 16:
                valid[5] = 0
                hand_action[5] = np.nan  # Missing label must never reach FK or training.
            rows.append(
                {
                    "episode_index": eid,
                    "frame_index": frame,
                    "timestamp": frame / fps,
                    "index": start + frame,
                    "task_index": 7,
                    "observation.state": body[::-1],
                    "action": (body + 0.1)[::-1],
                    "observation.hand_position": hands[::-1],
                    "hand_target": hand_action[::-1],
                    "hand_target_valid": valid[::-1],
                }
            )
    if version == "v3.0":
        ep_path = root / "meta/episodes/chunk-000/file-000.parquet"
        ep_path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(episodes), ep_path)
        task = pa.table({"task_index": [7], "__index_level_0__": ["synthetic task"]})
        task = task.replace_schema_metadata(
            {b"pandas": json.dumps({"index_columns": ["__index_level_0__"]}).encode()}
        )
        pq.write_table(task, root / "meta/tasks.parquet")
        # One episode crosses data-file boundaries; files also contain multiple episodes.
        for i, subset in enumerate([rows[:21], rows[21:47], rows[47:]]):
            p = root / f"data/chunk-000/file-{i:03d}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(subset), p)
    else:
        adapter.write_jsonl(root / "meta/episodes.jsonl", episodes)
        adapter.write_jsonl(
            root / "meta/tasks.jsonl", [{"task_index": 7, "task": "synthetic task"}]
        )
        for eid in ids:
            p = root / adapter.DATA_PATH.format(episode_chunk=0, episode_index=eid)
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([r for r in rows if r["episode_index"] == eid]), p)
    if videos:
        # Encode grayscale frames with monotonically increasing brightness.
        frames = np.broadcast_to(
            np.arange(60, dtype=np.uint8)[:, None, None, None] * 3, (60, 16, 16, 3)
        ).copy()
        for key in adapter.CAMERAS.values():
            chunks = (
                [(root / f"videos/{key}/chunk-000/file-000.mp4", frames)]
                if version == "v3.0"
                else [
                    (
                        root
                        / adapter.VIDEO_PATH.format(
                            episode_chunk=0, episode_index=eid, video_key=key
                        ),
                        clip,
                    )
                    for eid, clip in zip(ids, [frames[:40], frames[40:]])
                ]
            )
            for path, clip in chunks:
                path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        videos[0],
                        "-nostdin",
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        "16x16",
                        "-framerate",
                        str(fps),
                        "-i",
                        "pipe:0",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(path),
                    ],
                    input=clip.tobytes(),
                    check=True,
                    capture_output=True,
                )
    return adapter.Source(root)


def fake_fk(joints, joint_names, units):
    assert joint_names == adapter.JOINT_NAMES
    assert units == ["rad"] * 42
    assert np.isfinite(joints).all()
    poses = np.zeros((len(joints), 2, 7))
    poses[:, 0, :3] = joints[:, 8:11]
    poses[:, 1, :3] = joints[:, 15:18]
    poses[:, :, 6] = 1
    return poses


def provider():
    return SimpleNamespace(
        compute_poses=fake_fk,
        __file__=__file__,
        KINEMATICS={
            "frame": "synthetic",
            "left_tcp": "left",
            "right_tcp": "right",
            "model_sha256": "synthetic fixture, not robot FK",
        },
    )


def split():
    return {"train": [3], "validation": [9], "test": [], "exclude": []}


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_schema_order_shards_and_validity(tmp_path, version):
    source = make_source(tmp_path, version)
    report = adapter.inspect(source)
    assert report["episodes"][0]["usable_segments_16_frames"] == [(0, 16), (17, 40)]
    assert report["episodes"][0]["all_hand_targets_valid_frames"] == 39
    state, action, valid, tasks, indices = source.episode(source.episodes[0])
    np.testing.assert_allclose(state[0, :22], np.arange(22) / 100)
    np.testing.assert_allclose(action[0, 22:], np.arange(20) / 100 + 0.1)
    assert not valid[16] and tasks[0] == 7 and indices[-1] == 39


@pytest.mark.parametrize("version", ["v2.1", "v3.0"])
def test_full_conversion_matches_labels_and_video_frames(tmp_path, calibration, binaries, version):
    source = make_source(tmp_path / "source", version, binaries)
    before = {p: adapter.sha256(p) for p in source.root.rglob("*") if p.is_file()}
    output = tmp_path / "prepared"
    report = adapter.prepare(
        source,
        output,
        calibration,
        split(),
        ["joints", "eef"],
        "synthetic-revision",
        provider(),
        ffmpeg=binaries[0],
        ffprobe=binaries[1],
    )
    assert report["output_frames"] == {"train": 39, "validation": 20, "test": 0}
    assert report["episodes"][0]["invalid_hand_frames"] == 1
    assert all(adapter.sha256(p) == digest for p, digest in before.items())
    for variant, size in [("joints", 42), ("eef", 40)]:
        root = output / variant / "train"
        info = adapter.read_json(root / "meta/info.json")
        assert info["features"]["action"]["shape"] == [size]
        assert info["total_episodes"] == 2
        assert not (root / "meta/stats.json").exists()
        table = pq.read_table(root / "data/chunk-000/episode_000001.parquet")
        assert table["source_frame_index"].to_pylist() == list(range(17, 40))
        assert table["timestamp"][0].as_py() == 0
        assert table["index"][0].as_py() == 16
        assert table[adapter.ANNOTATION].to_pylist() == [7] * 23
        obs, action = (
            np.array(table["observation.state"].to_pylist()),
            np.array(table["action"].to_pylist()),
        )
        np.testing.assert_allclose(obs[0, :8], (np.arange(8) / 100 + 0.017) * 2 + 0.1, atol=1e-6)
        np.testing.assert_allclose(action[0, :8], (np.arange(8) / 100 + 0.117) * 3 - 0.2, atol=1e-6)
        if variant == "eef":
            np.testing.assert_allclose(
                action[0, 8:11], (np.arange(8, 11) / 100 + 0.117) * 3 - 0.2, atol=1e-6
            )
            np.testing.assert_allclose(action[:, 11:14], 0)
        for split_name, eid, expected_start, count in [
            ("train", 1, 17, 23),
            ("validation", 0, 40, 20),
        ]:
            video = (
                output
                / variant
                / split_name
                / "videos/chunk-000/observation.images.head_stereo_left"
                / f"episode_{eid:06d}.mp4"
            )
            decoded = subprocess.run(
                [
                    binaries[0],
                    "-v",
                    "error",
                    "-i",
                    str(video),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            ).stdout
            frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, 16, 16, 3)
            assert len(frames) == count
            np.testing.assert_allclose(
                frames.mean(axis=(1, 2, 3)),
                np.arange(expected_start, expected_start + count) * 3,
                atol=5,
            )


def test_splits_require_complete_assignment_and_recording_isolation(tmp_path):
    source = make_source(tmp_path)
    for manifest in [
        {"train": [3]},
        {"train": [3, 9], "test": [9]},
        {**split(), "recording_groups": {"3": "same-take", "9": "same-take"}},
    ]:
        with pytest.raises(ValueError):
            adapter.split_assignments(manifest, source.episodes)


def test_calibration_and_fk_fail_closed(calibration):
    bad = deepcopy(calibration)
    bad["joints"][adapter.JOINT_NAMES[0]]["unit"] = None
    with pytest.raises(ValueError, match="unit"):
        adapter.calibration_arrays(bad)
    bad = provider()
    bad.compute_poses = lambda *args: np.zeros((2, 2, 7))
    with pytest.raises(ValueError, match="unit quaternions"):
        adapter.eef_values(np.zeros((2, 42)), bad, ["rad"] * 42)
    bad.compute_poses = lambda *args: np.zeros((2, 14))
    with pytest.raises(ValueError, match="FK must return"):
        adapter.eef_values(np.zeros((2, 42)), bad, ["rad"] * 42)


def test_invalid_timing_and_nan_labels_are_rejected(tmp_path):
    source = make_source(tmp_path)
    path = tmp_path / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1]["timestamp"] += 0.01
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="timestamps"):
        adapter.Source(tmp_path).episode(source.episodes[0])
    rows[1]["timestamp"] -= 0.01
    rows[0]["hand_target"][0] = float("nan")
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="nonfinite valid hand"):
        adapter.Source(tmp_path).episode(source.episodes[0])


def test_no_overwrite_and_failed_build_cleanup(tmp_path, calibration, binaries):
    source = make_source(tmp_path / "source")  # Deliberately missing videos.
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Missing video"):
        adapter.prepare(
            source,
            output,
            calibration,
            split(),
            ["joints"],
            "test",
            ffmpeg=binaries[0],
            ffprobe=binaries[1],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared-*"))
    output.mkdir()
    (output / "keep").write_text("user data")
    with pytest.raises(ValueError, match="Output must be new"):
        adapter.prepare(source, output, calibration, split(), ["joints"], "test")
    assert (output / "keep").read_text() == "user data"
    with pytest.raises(ValueError, match="calibrated FK"):
        adapter.prepare(source, tmp_path / "other", calibration, split(), ["eef"], "test")


def test_short_spans_are_not_joined_across_gaps():
    valid = np.r_[np.ones(10, dtype=bool), False, np.ones(10, dtype=bool)]
    assert adapter.valid_segments(valid, 16) == []


def test_quaternion_orientation_and_non_arm_targets():
    model = provider()

    def rotated(joints, names, units):
        poses = fake_fk(joints, names, units)
        poses[:, :, 3:] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        poses[:, 1, 3:] *= -1  # q and -q describe the same orientation.
        return poses

    model.compute_poses = rotated
    joints = np.arange(84).reshape(2, 42) / 100
    result = adapter.eef_values(joints, model, ["rad"] * 42)
    np.testing.assert_allclose(result[:, :8], joints[:, :8])
    np.testing.assert_allclose(result[:, 20:], joints[:, 22:])
    np.testing.assert_allclose(result[:, 11:14], [[0, 0, np.pi / 2]] * 2)
    np.testing.assert_allclose(result[:, 17:20], [[0, 0, np.pi / 2]] * 2)


def test_schema_changes_and_global_index_mismatch_fail(tmp_path):
    source = make_source(tmp_path)
    path = tmp_path / "data/chunk-000/file-000.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[1]["index"] = 100
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="global frame indices"):
        adapter.Source(tmp_path).episode(source.episodes[0])
    info = deepcopy(source.info)
    info["features"]["action"]["names"][0] = "unknown_joint"
    adapter.write_json(tmp_path / "meta/info.json", info)
    with pytest.raises(ValueError, match="named G2 joints"):
        adapter.Source(tmp_path)


def test_source_mutation_aborts_and_removes_staging(tmp_path, calibration, binaries, monkeypatch):
    source = make_source(tmp_path / "source", videos=binaries)
    output = tmp_path / "prepared"
    original_cut = adapter.cut_video

    def cut_and_mutate(*args):
        result = original_cut(*args)
        with (source.root / "meta/info.json").open("a") as stream:
            stream.write("\n")
        return result

    monkeypatch.setattr(adapter, "cut_video", cut_and_mutate)
    adapter_split = {"train": [9], "exclude": [3]}
    with pytest.raises(ValueError, match="Source files changed"):
        adapter.prepare(
            source,
            output,
            calibration,
            adapter_split,
            ["joints"],
            "test",
            ffmpeg=binaries[0],
            ffprobe=binaries[1],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared-*"))
