# SPDX-License-Identifier: Apache-2.0
"""Training orchestration tests do not launch models or require GPUs."""

import json
import subprocess

import pytest
from scripts.agibot_g2 import run_training as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "prepared"
    write(root / "preparation_manifest.json", {"variants": ["joints", "eef"]})
    for variant, size in [("joints", 42), ("eef", 40)]:
        for split in ("train", "validation", "test"):
            meta = root / variant / split / "meta"
            write(
                meta / "info.json",
                {
                    "features": {k: {"shape": [size]} for k in ("observation.state", "action")},
                    "total_episodes": 1,
                },
            )
            write(meta / "episodes.jsonl", {"episode_index": 0, "length": 123})
            write(meta / "modality.json", {})
            write(meta / "tasks.jsonl", {"task_index": 0, "task": "synthetic"})
            for name in runner.STATS_FILES:
                write(meta / name, {"synthetic": [1]})
    return root


def args(root, tmp_path, stage, *extra):
    return runner.parse_args(
        [
            stage,
            "--prepared-root",
            str(root),
            "--variant",
            "joints",
            "--run-dir",
            str(tmp_path / stage),
            *extra,
        ]
    )


def test_stats_uses_training_only_and_copies_heldout(prepared, tmp_path, monkeypatch):
    calls = []
    for split in ("validation", "test"):
        for name in runner.STATS_FILES:
            write(prepared / "joints" / split / "meta" / name, {"stale": [0]})
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    a = args(prepared, tmp_path, "stats")
    runner.execute(a)
    assert len(calls) == 1
    assert calls[0][calls[0].index("--dataset-path") + 1] == str(prepared / "joints/train")
    result = runner.read_json(a.run_dir / "run.json")
    assert result["status"] == "completed"
    assert len(result["statistics_copies"]) == 4
    assert set(result["generated_statistics"]) == set(runner.STATS_FILES)
    for split in ("validation", "test"):
        runner.check_stats(prepared / "joints/train", prepared / "joints" / split)


@pytest.mark.parametrize("stage,steps", [("smoke", "20"), ("train", "2000")])
def test_training_plan_and_dry_run(prepared, tmp_path, stage, steps, monkeypatch):
    a = args(prepared, tmp_path, stage, "--dry-run")
    commands, copies, hashes = runner.plan(a)
    command = commands[0]
    assert command[command.index("--max-steps") + 1] == steps
    assert command[command.index("--episode-sampling-rate") + 1] == "1.0"
    assert "--skip-weight-loading" not in command
    assert "preparation_manifest" in hashes and not copies
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("dry run launched a subprocess")
    )
    runner.execute(a)
    assert not a.run_dir.exists()


def test_eval_uses_entire_heldout_episode_and_unique_plot(prepared, tmp_path):
    checkpoint = tmp_path / "checkpoint-2000"
    checkpoint.mkdir()
    a = args(prepared, tmp_path, "eval", "--checkpoint", str(checkpoint))
    commands, _, _ = runner.plan(a)
    c = commands[0]
    assert c[c.index("--dataset-path") + 1] == str(prepared / "joints/validation")
    assert c[c.index("--steps") + 1] == "123"
    assert c[c.index("--save-plot-path") + 1].endswith("episode_000000.png")
    write(prepared / "joints/validation/meta/stats.json", {"wrong": [2]})
    with pytest.raises(ValueError, match="match training statistics"):
        runner.plan(a)


def test_existing_run_missing_stats_and_wrong_dimensions_fail(prepared, tmp_path):
    a = args(prepared, tmp_path, "train")
    a.run_dir.mkdir()
    with pytest.raises(ValueError, match="run-dir must be new"):
        runner.execute(a)
    (prepared / "joints/train/meta/stats.json").unlink()
    with pytest.raises(ValueError, match="stats stage first"):
        runner.plan(a)
    write(
        prepared / "joints/train/meta/info.json",
        {"features": {"observation.state": {"shape": [40]}}},
    )
    with pytest.raises(ValueError, match="42 values"):
        runner.plan(a)


def test_failed_training_preserves_failed_run_record(prepared, tmp_path, monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    a = args(prepared, tmp_path, "train")
    with pytest.raises(subprocess.CalledProcessError):
        runner.execute(a)
    record = runner.read_json(a.run_dir / "run.json")
    assert record["status"] == "failed"
    assert (a.run_dir / "command_000.log").exists()
