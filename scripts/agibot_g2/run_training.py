# SPDX-License-Identifier: Apache-2.0
"""Run GR00T statistics, smoke training, fine-tuning, or held-out evaluation for G2.

Use the activated GR00T Python environment. --dry-run prints the commands without
starting jobs or writing files. This wrapper does not install dependencies.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
STATS_FILES = ("stats.json", "relative_stats.json")


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(root, variant, split):
    path = root / variant / split
    info = read_json(path / "meta/info.json")
    size = 42 if variant == "joints" else 40
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [size]:
            raise ValueError(f"{path}: {key} must have {size} values")
    episodes = [
        json.loads(line) for line in (path / "meta/episodes.jsonl").read_text().splitlines()
    ]
    if not episodes or [e["episode_index"] for e in episodes] != list(range(len(episodes))):
        raise ValueError(f"{path}: expected nonempty consecutive prepared episode IDs")
    if info["total_episodes"] != len(episodes):
        raise ValueError(f"{path}: inconsistent episode count")
    for name in ("modality.json", "tasks.jsonl"):
        if not (path / "meta" / name).is_file():
            raise ValueError(f"Missing {path / 'meta' / name}")
    return path, episodes


def check_stats(train, heldout=None):
    for name in STATS_FILES:
        path = train / "meta" / name
        if not path.is_file() or not read_json(path):
            raise ValueError(f"Missing/empty {path}; run the stats stage first")
        if heldout is not None and (
            not (heldout / "meta" / name).is_file()
            or digest(path) != digest(heldout / "meta" / name)
        ):
            raise ValueError(f"{heldout}: {name} must match training statistics; run stats first")


def plan(args):
    root = args.prepared_root.resolve()
    manifest_path = root / "preparation_manifest.json"
    manifest = read_json(manifest_path)
    if args.variant not in manifest["variants"]:
        raise ValueError(f"{args.variant} was not prepared in this dataset root")
    train, _ = dataset(root, args.variant, "train")
    config = REPO / f"examples/AGIBOT_G2/{args.variant}_config.py"
    commands, copies = [], []
    common = ["--embodiment-tag", "NEW_EMBODIMENT"]
    if args.stage == "stats":
        commands.append(
            [
                sys.executable,
                str(REPO / "gr00t/data/stats.py"),
                "--dataset-path",
                str(train),
                *common,
                "--modality-config-path",
                str(config),
            ]
        )
        for split in ("validation", "test"):
            if (root / args.variant / split).exists():
                heldout, _ = dataset(root, args.variant, split)
                copies.extend(
                    (train / "meta" / name, heldout / "meta" / name) for name in STATS_FILES
                )
    elif args.stage in ("smoke", "train"):
        check_stats(train)
        if args.num_gpus < 1 or args.batch_size < 1 or args.batch_size % args.num_gpus:
            raise ValueError("Batch size must be positive and divisible by num-gpus")
        if args.workers < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("workers must be nonnegative and learning-rate positive")
        steps = (
            args.max_steps
            if args.max_steps is not None
            else (20 if args.stage == "smoke" else 2000)
        )
        if steps < 1:
            raise ValueError("max-steps must be positive")
        save_steps = min(steps, 20 if args.stage == "smoke" else 500)
        commands.append(
            [
                sys.executable,
                str(REPO / "gr00t/experiment/launch_finetune.py"),
                "--base-model-path",
                args.base_model,
                "--dataset-path",
                str(train),
                *common,
                "--modality-config-path",
                str(config),
                "--output-dir",
                str(args.run_dir / "checkpoints"),
                "--num-gpus",
                str(args.num_gpus),
                "--global-batch-size",
                str(args.batch_size),
                "--max-steps",
                str(steps),
                "--save-steps",
                str(save_steps),
                "--save-total-limit",
                "4",
                "--dataloader-num-workers",
                str(args.workers),
                "--learning-rate",
                str(args.learning_rate),
                "--episode-sampling-rate",
                "1.0",
            ]
        )
    else:
        if args.checkpoint is None or not args.checkpoint.is_dir():
            raise ValueError("eval requires --checkpoint pointing to an existing local checkpoint")
        heldout, episodes = dataset(root, args.variant, args.split)
        check_stats(train, heldout)
        for episode in episodes:
            eid = episode["episode_index"]
            commands.append(
                [
                    sys.executable,
                    str(REPO / "gr00t/eval/open_loop_eval.py"),
                    "--dataset-path",
                    str(heldout),
                    *common,
                    "--model-path",
                    str(args.checkpoint.resolve()),
                    "--traj-ids",
                    str(eid),
                    "--execution-horizon",
                    "16",
                    "--steps",
                    str(episode["length"]),
                    "--save-plot-path",
                    str(args.run_dir / f"episode_{eid:06d}.png"),
                ]
            )
    hashes = {
        "preparation_manifest": digest(manifest_path),
        "modality_config": digest(config),
        "training_info": digest(train / "meta/info.json"),
        "training_modality": digest(train / "meta/modality.json"),
    }
    for name in STATS_FILES:
        path = train / "meta" / name
        if path.exists():
            hashes[name] = digest(path)
    return commands, copies, hashes


def execute(args):
    if args.run_dir is None:
        raise ValueError("--run-dir is required; use a fresh path for each stage and variant")
    args.run_dir = args.run_dir.resolve()
    root = args.prepared_root.resolve()
    if (
        args.run_dir.exists()
        or args.run_dir.is_relative_to(root)
        or root.is_relative_to(args.run_dir)
    ):
        raise ValueError("run-dir must be new and outside the prepared dataset")
    commands, copies, hashes = plan(args)
    for command in commands:
        print(shlex.join(command), flush=True)
    for source, destination in copies:
        print(f"Copy training statistics: {source} -> {destination}", flush=True)
    if args.dry_run:
        return
    args.run_dir.mkdir(parents=True)
    record = {
        "stage": args.stage,
        "variant": args.variant,
        "prepared_root": str(root),
        "python": sys.executable,
        "runner_sha256": digest(Path(__file__)),
        "input_hashes": hashes,
        "commands": commands,
        "statistics_copies": [[str(a), str(b)] for a, b in copies],
        "status": "running",
    }
    manifest = args.run_dir / "run.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for index, command in enumerate(commands):
            log_path = args.run_dir / f"command_{index:03d}.log"
            print(f"Running command {index + 1}/{len(commands)}; log: {log_path}", flush=True)
            with log_path.open("w") as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.stage == "stats":
            train = root / args.variant / "train"
            check_stats(train)
            for source, destination in copies:
                shutil.copyfile(source, destination)
            record["generated_statistics"] = {
                name: digest(train / "meta" / name) for name in STATS_FILES
            }
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stats", "smoke", "train", "eval"])
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--variant", choices=["joints", "eef"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    execute(parse_args())
