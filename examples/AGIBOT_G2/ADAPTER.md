# Preparing the G2 training datasets

[`prepare_dataset.py`](../../scripts/agibot_g2/prepare_dataset.py) accepts a local
LeRobot v3.0, v2.1, or v2.0 G2 snapshot and writes GR00T v2.1 datasets in a new
output directory. It needs no model weights, GPU, or LeRobot installation.
It supports the named 22 body/head/arm joints, 20 hand joints, and three RGB
cameras described in the [training guide](README.md).

The joint variant is implemented. The EEF variant is implemented against an
explicit FK-provider interface: it needs your calibrated G2 kinematics to
produce real pose targets. It never treats recorded observed poses or VR poses
as commanded robot poses. Synthetic FK tests establish the adapter contract,
not the accuracy of G2 kinematics.

## 1. Inspect the completed snapshot

Download a fixed Hugging Face revision to a new directory once the upload is
complete. Do not run conversion against a directory that is still changing.
The adapter takes a local source; it does not download or publish datasets.

If `meta/episode_flags.json` is present, preparation requires each selected
episode to have status `accepted`; rejected episodes are filtered automatically.
Use `--max-episode-index 64` for an inclusive episode-ID cutoff. The intended
selection is `episode_index <= 64` and review status `accepted`.

From the repository root:

```bash
uv run --script scripts/agibot_g2/prepare_dataset.py inspect \
  --source /data/g2-source > /data/g2-inspection.json
```

`uv` supplies the script's Python 3.12+ CPU dependencies in an isolated script
environment. Alternatively, install NumPy, PyArrow, and SciPy in your own Python
3.12 environment and run the script with `python`. Video preparation also needs
`ffmpeg` with `libx264` and `ffprobe` on PATH; explicit executable paths can be
passed with `--ffmpeg` and `--ffprobe`.

Inspection reads numerical rows and metadata, without requiring video files.
It checks joint names and dimensions, episode lengths, frame indices, timestamps,
task references, finite feedback/command values, and binary hand-validity flags.
It reports, per episode, how many rows have all twenty hand targets available
and the contiguous usable intervals of at least sixteen frames.

Export the two templates from that report:

```bash
python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('/data/g2-inspection.json').read_text())
for key, filename in [('calibration_template', 'g2-calibration.json'),
                      ('split_template', 'g2-splits.json')]:
    path = Path('/data') / filename
    with path.open('x') as stream:
        json.dump(report[key], stream, indent=2)
        stream.write('\n')
PY
```

This intentionally refuses to overwrite existing calibration/split files.

## 2. Fill in joint calibration and choose splits

Calibration has an entry for **every named joint**, for example:

```json
{
  "joints": {
    "idx21_arm_l_joint1": {
      "unit": "rad",
      "state": {"scale": 1.0, "offset": 0.0},
      "action": {"scale": 1.0, "offset": 0.0}
    }
  }
}
```

This is an example entry, not a complete calibration file. Fill in all 42
entries from the generated template. The rule for each signal is:

```text
prepared_value = source_value * scale + offset
```

Use `unit: "rad"` for revolute joints and `unit: "m"` for prismatic joints.
A scale of one is appropriate only after confirming that the source already
uses the desired units and sign. Degrees-to-radians uses `pi / 180`; offsets
are expressed in the output units. Feedback and command scales/offsets are
separate because their source encodings may differ, particularly for hands.
If a signal requires a nonlinear mapping, implement that conversion first;
this adapter only implements affine per-joint calibration. Null template
values are rejected rather than interpreted as identity calibration.

The split file assigns **selected source episode IDs** (after review-flag and
maximum-ID filtering):

```json
{
  "train": [0, 1, 2],
  "validation": [3],
  "test": [4],
  "exclude": [5],
  "recording_groups": {"0": "take-a", "1": "take-a"}
}
```

Use the IDs in your snapshot, not these illustrative IDs. Every episode must
appear exactly once, including episodes deliberately excluded after review.
An updated dataset with unassigned episodes causes an error instead of silently
putting them into training. The generated template initially assigns everything
to train for a smoke run; edit it to create held-out splits for a real experiment.
An empty validation/test list is permitted and creates no directory for that
split. A requested split with no usable segments causes preparation to fail.

All segments from an episode inherit that episode's split. If several source
episodes came from the same recording, assign their original recording identity
through `recording_groups`; the adapter rejects groups spanning train/validation/
test. Without that optional mapping, each source episode is treated as a distinct
recording. The adapter cannot infer success labels or original take relationships
that are absent from the supplied inputs.

## 3. Prepare joint prediction

```bash
uv run --script scripts/agibot_g2/prepare_dataset.py prepare \
  --source /data/g2-source \
  --output /data/g2-prepared \
  --source-revision YOUR_EXACT_HF_COMMIT_SHA \
  --calibration /data/g2-calibration.json \
  --splits /data/g2-splits.json \
  --variants joints \
  --max-episode-index 64
```

Replace the revision placeholder with the commit used for the download. This
is a caller-supplied provenance label, not an online revision verification.
The manifest additionally records SHA-256 hashes of source data, metadata, and
selected videos. These are checked before and after preparation to detect
changes during conversion.

The adapter:

- Reorders each source signal by joint name and applies explicit calibration.
- Builds 42-value state and action vectors including both hands.
- Drops rows with any invalid hand target and splits at every gap. Missing
  targets are never replaced with zero or fed into FK. Valid spans shorter than
  `--min-segment-frames` (default 16, minimum 16) are discarded. The manifest
  reports both invalid rows and valid rows lost because their spans were short.
- Reads episodes even when their rows span multiple v3 Parquet files.
- Cuts each camera video to exactly the retained frame range, decoding and
  re-encoding H.264 rather than copying from an approximate keyframe. This adds
  a lossy encoding generation and can take substantial CPU time.
- Validates source and output video frame timelines/counts, frame rates, and
  source resolution metadata. Videos must have one encoded frame per dataset
  timestep and a zero-based constant-rate timeline. This checks alignment in
  the exported dataset; it does not calibrate camera exposure or network delays.
- Renumbers output episodes/frames, resets segment timestamps to zero, and
  writes the task annotation and complete `info.json`/`modality.json` metadata.

Output structure:

```text
/data/g2-prepared/
  preparation_manifest.json
  joints/
    train/{data,videos,meta}/
    validation/{data,videos,meta}/
    test/{data,videos,meta}/
```

Original source episode/frame/global indices are retained as
`source_episode_index`, `source_frame_index`, and `source_global_index`. The
manifest records segment bounds with an exclusive end frame, split assignments,
calibration, source hashes, and the adapter hash. Auxiliary source streams are
not copied into the training view; use the immutable source snapshot and this
mapping to inspect them. No source statistics are copied.

The output path must not exist and must be outside the source tree. Work is
staged beside the destination and published only after all checks succeed.
Failed builds remove their own staging directory. Source files are never
modified. Existing prepared data is never overwritten.

## 4. Prepare EEF + joint prediction

Implement a trusted local Python module, for example `/data/g2_fk.py`, with:

```python
KINEMATICS = {
    "frame": "your_fixed_robot_root_frame",
    "left_tcp": "your_calibrated_left_tcp",
    "right_tcp": "your_calibrated_right_tcp",
    "model_sha256": "SHA256_OF_YOUR_CALIBRATED_ROBOT_DESCRIPTION",
}


def compute_poses(joints, joint_names, units):
    """Return shape (N, 2, 7): left/right [x, y, z, qx, qy, qz, qw].

    joints is (N, 42), already calibrated; joint_names gives its exact order.
    units gives rad/m per joint. Include all body joints affecting the arms.
    Return metre positions and unit XYZW quaternions in the documented frame,
    with the documented TCP offsets applied.
    """
    raise NotImplementedError("Connect and validate the calibrated G2 FK here")
```

That snippet documents the interface; it is not a working G2 FK implementation.
The provider is ordinary Python loaded and executed locally, so use a file you
trust. Its metadata and source hash are included in the preparation manifest.
The adapter checks returned shapes, finiteness, and unit quaternions; confirming
physical correctness of its frame, geometry, and TCP definitions requires a
separate FK validation against the robot/recordings.

Then generate both variants together into a **new** destination:

```bash
uv run --script scripts/agibot_g2/prepare_dataset.py prepare \
  --source /data/g2-source \
  --output /data/g2-prepared-both \
  --source-revision YOUR_EXACT_HF_COMMIT_SHA \
  --calibration /data/g2-calibration.json \
  --splits /data/g2-splits.json \
  --variants joints eef \
  --fk-provider /data/g2_fk.py
```

FK is evaluated separately on measured joints for state and commanded joints
for action. The EEF variant replaces the fourteen arm joints with two
`XYZ_ROTVEC` poses and retains all 28 body/head/hand joint targets, for **40
values per timestep**. Absolute poses/targets remain in the prepared dataset;
GR00T handles relative arm actions through the existing modality configurations.
Both variants use exactly the same segments, splits, task labels, and video
frames. The second variant receives copies of the already-encoded videos.

Future-observation pose supervision, learned validity masks, nonlinear hand
calibration, automatic IK, and frame-rate resampling are outside this adapter.
The current path implements command imitation with FK-derived EEF labels.

## 5. Generate statistics, then train

The adapter does not compute GR00T statistics or start training. The
[training entry point and checklist](TRAINING.md) cover statistics, smoke runs,
fine-tuning, and held-out evaluation. You can also use the
[training commands](README.md#gr00t-metadata-and-training) from the main GR00T
Python environment after inspecting the prepared data. Set the dataset path to
`.../joints/train` or `.../eef/train` and use the matching modality config.
Generate statistics on **training only**. For standalone held-out dataset loading,
copy the generated `meta/stats.json` and `meta/relative_stats.json` from that
variant's training root into its validation/test roots; model inference uses
the training/checkpoint normalization. Never mix statistics between variants.

Inspect the preparation manifest before launching: many discarded hand-label
rows can remove entire task phases. A successful file conversion alone does
not establish that the remaining demonstrations are sufficient to learn the
complete task.

## CPU verification

The synthetic integration tests cover v2/v3, source joint reordering, episodes
spanning files, invalid hand gaps, distinct feedback/command calibration,
EEF-provider labels, split isolation, output metadata, exact retained video
frames, and source preservation. No private recordings or robot models are
included in the tests.

```bash
uv run --no-project --python 3.12 \
  --with numpy --with pyarrow --with scipy --with pytest --with filelock \
  python -m pytest tests/scripts/test_agibot_g2_adapter.py -q
```

Video tests require `ffmpeg` and `ffprobe`; otherwise pytest reports them as
skipped. `G2_TEST_FFMPEG` and `G2_TEST_FFPROBE` can specify their executable paths.
