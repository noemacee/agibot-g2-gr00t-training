# AGIBOT G2: joint targets and end-effector pose targets

This fork adapts the training workflow of NVIDIA Isaac GR00T N1.7 to AGIBOT G2.
It provides a training design, two modality configurations, and a
[dataset adapter](ADAPTER.md) that writes both training layouts from LeRobot v2/v3
snapshots. **G2 kinematic calibration, training runs, and robot controller
integration remain to be completed.** The EEF adapter requires a calibrated FK
provider. Follow the adapter instructions before launching either training run.

Upstream baseline: `NVIDIA/Isaac-GR00T` at
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
Dataset: [noemacee/g2-bag-to-bin-2026-09-13](https://huggingface.co/datasets/noemacee/g2-bag-to-bin-2026-09-13),
revision `09ef91fc1de8a4eddc0f75c96c4e12b02537a230` inspected on 2026-09-13.
The dataset requires authorized Hugging Face access. Keep downloaded data,
provenance, and training outputs outside this public repository.

See [training status, runnable stages, and remaining checklist](TRAINING.md)
for everything needed between this PR and trained/evaluated policies.

## What the recordings provide

The dataset card and `meta/info.json` describe 31 episodes, 63,269 frames,
60 Hz sampling, and LeRobot v3.0. Task-content review does not establish that
every episode succeeded. The following is a metadata inspection, not a fresh
validation of all trajectories or videos.

| Source | Meaning and dimensions |
| --- | --- |
| `observation.state` | 22 body/head/arm joint feedback values |
| `action` | 22 body/head/arm joint command values |
| `observation.hand_position` | 20 hand feedback values |
| `hand_target`, `hand_target_valid` | 20 hand targets and per-joint validity |
| `observation.end_effector_pose` | Left then right `[x,y,z,qx,qy,qz,qw]`, 14 values |
| `observation.images.head_stereo_left` | Head RGB camera |
| `observation.images.hand_left_color` | Left wrist RGB camera |
| `observation.images.hand_right_color` | Right wrist RGB camera |
| `task_index` | Index into the task table |

The 22-joint ordering is body `[0:5]`, head `[5:8]`, left arm `[8:15]`,
and right arm `[15:22]`. Hands are left `[0:10]`, right `[10:20]` in their
separate columns. Verify exact feature names against the pinned metadata;
do not assume that URDF or controller joint order matches this order.

Joint values preserve source units. Establish units, sign, offsets, and limits
per joint before converting to a controller or FK representation. Pose field
names establish quaternion order, but do not establish the coordinate frame,
length units, TCP definition, or whether the derived stream is measured FK or
a teleoperation target. Confirm those from the recorder and G2 model.

Sampling uses causal collector receipt times. Source camera images are roughly
30 Hz and repeat on the 60 Hz grid. Exposure and transport delays are not
calibrated. Retain source-age and provenance information for local validation;
do not mistake repeated frames for independent visual observations.

## Two experiments

Both models predict the complete set of control targets represented in this
dataset. Body, head, and both hands are learned joint outputs in **both**
variants, not just observations or externally supplied commands:

- **Joint model:** 5 body + 3 head + 7 left arm + 7 right arm + 10 left hand
  + 10 right hand = **42 joint targets per timestep**.
- **EEF + joint model:** left and right arm poses (6 values each in the chosen
  XYZ + rotation-vector format) **plus 28 joint targets** (5 body + 3 head
  + 10 left hand + 10 right hand) = **40 action values per timestep**.

The EEF variant changes only the representation of the two arms. It predicts
both arm poses and the other joints together in one synchronized action chunk
from one model, with supervision for every included output. Arm joint commands
are then obtained through IK; body, head, and hand joint targets come directly
from the model. The scope here is the joints recorded in this dataset; any
additional robot actuators require corresponding data and configuration.

Train separate checkpoints initially, with identical source episode splits,
camera views, instructions, action horizon, and training budget.

| | Joint prediction | End-effector prediction |
| --- | --- | --- |
| Arm outputs | Two sets of seven joint targets | Two Cartesian poses |
| Main labels | Recorded joint commands | FK of commanded joints, or explicitly shifted observed poses |
| Hands | Twenty absolute hand targets | Same twenty absolute hand targets |
| Body/head | Eight absolute joint targets | Same eight absolute joint targets |
| Robot execution | Joint position controller | IK/Cartesian controller plus body/head/hand control |
| Main prerequisite | Joint mapping and valid command labels | Calibrated kinematics, pose frames, TCP, and IK |

Start with joint prediction to establish the data-loading and imitation baseline.
Then compare relative EEF prediction, which matches the action-space approach
used by N1.7 pretraining. This is an experimental choice, not evidence that one
will perform better on this dataset. Predicting arm joint targets in addition
to arm poses would be a separate experiment requiring a clear choice of which
arm targets the controller executes. The EEF + joint model described here
already combines arm pose prediction with joint prediction for the rest of
the recorded robot.

## Shared data preparation

Use the [adapter commands](ADAPTER.md) for the implemented workflow. The adapter
reads v3 directly, so the standalone upstream conversion in step 2 below is an
alternative, not a prerequisite. Steps 3–4 describe operations the adapter now
performs; unit calibration and split selection are explicit user inputs.

1. Download a pinned snapshot using existing Hugging Face login credentials:

   ```bash
   hf download noemacee/g2-bag-to-bin-2026-09-13 \
     --repo-type dataset --revision 09ef91fc1de8a4eddc0f75c96c4e12b02537a230 \
     --local-dir /data/g2-conversion/noemacee/g2-bag-to-bin-2026-09-13
   ```

2. Convert that disposable working copy to LeRobot v2.1. The upstream converter
   renames its input and can delete existing sibling `_v30`/`_v2.1` folders;
   use a fresh conversion directory for each run. It preserves auxiliary data
   columns, but does not create the G2 training layout described below.
   Its separate environment requires Python **3.10 or 3.11**, while main GR00T
   uses Python **3.12**:

   ```bash
   (
     cd scripts/lerobot_conversion
     uv venv --python 3.11
     source .venv/bin/activate
     uv pip install -e .
     python convert_v3_to_v2.py \
       --repo-id noemacee/g2-bag-to-bin-2026-09-13 \
       --root /data/g2-conversion
   )
   ```

3. Run the G2 adapter to create separate `joints` and `eef` dataset roots
   with the layouts below. Store float32 arrays in `observation.state` and
   `action`; update feature shapes/names in `meta/info.json`. Preserve timing,
   videos, task labels, and a mapping back to original episodes. Add an integer
   `annotation.human.task_description` column copied from `task_index` and the
   matching feature metadata. Export the v2 task and episode JSONL files.

4. Audit `hand_target_valid` before constructing action labels. A zero target
   with validity zero is missing data, not necessarily a hand-closing command.
   The adapter keeps these rows by default and imputes invalid hand dimensions
   with the measured hand position (hold), recording the affected rows in the
   preparation manifest. This is the requested training policy, but it is an
   assumption about missing commands and must be reviewed. Use
   `--drop-invalid-hand-targets` for strict filtering, or implement a documented
   per-timestep/per-dimension loss mask if the hold assumption is unsuitable.

5. Split by original recording before segmentation. A starting split is
   23 train / 4 validation / 4 test episodes, adjusted for success coverage and
   recording-session grouping. Keep all segments from a recording together.
   Review complete videos for success/failure and exclude or explicitly model
   unsuitable demonstrations. Record the exact split IDs and random seed.
   Materialize separate train/validation/test roots; do not rely on the source
   dataset's all-training split to provide held-out evaluation.

6. Start at the recorded 60 Hz with a 16-step chunk (about 0.267 seconds of
   control intervals). Keep offsets `[0,...,15]` for command labels. A later
   30 Hz experiment must resample all signals, videos, timestamps, and metadata
   consistently, and use a comparable horizon in seconds. Regenerate statistics
   after any layout, unit, sampling, or horizon change, using training data only.

## A. Joint-angle target layout

Construct both state and action as 42 values:

| Key | Slice | State | Action |
| --- | --- | --- | --- |
| `body` | `[0:5]` | Body feedback | Body commands |
| `head` | `[5:8]` | Head feedback | Head commands |
| `left_arm` | `[8:15]` | Left arm feedback | Left arm commands |
| `right_arm` | `[15:22]` | Right arm feedback | Right arm commands |
| `left_hand` | `[22:32]` | Left hand feedback | Valid left hand targets |
| `right_hand` | `[32:42]` | Right hand feedback | Valid right hand targets |

The first 22 values come from the original state/action arrays; append the
appropriate hand columns after validity filtering. Convert angular joints to
radians once the source units are established; retain proper linear units for
any prismatic joints. Never assume every body degree of freedom is angular.

Use [joints_config.py](joints_config.py). Arm actions are `NON_EEF`, `DEFAULT`,
and `RELATIVE`: the processor learns `q_command[t+k] - q_feedback[t]`.
Body/head/hands are absolute. Store **absolute** targets in the dataset; the
processor computes the relative representation. An all-absolute joint model
is a useful ablation, requiring its own statistics and checkpoint.

## B. End-effector pose target layout

The source pose column is an observation, not an explicit EEF command label.
Choose and record one target construction:

- **Preferred for comparing action representations:** obtain the calibrated G2
  URDF, named joint map, and TCP offsets. Compute the current TCP pose with
  `FK(q_feedback[t])` and the target TCP pose with `FK(q_command[t])` for each
  arm. Include body joints affecting each arm and use a consistent fixed robot
  root frame. Verify FK feedback against the recorded pose stream after any
  required frame transformation. This makes joint and EEF labels correspond
  to the same underlying command trajectory.
- **Alternative trajectory imitation:** after verifying the pose stream's
  semantics, use observed pose at `t+L` as the target at row `t`, where `L>0`
  is an explicit, validated lead in samples. Shift the other action labels to
  the same target time and drop terminal rows without a target. If comparing
  against joints, construct a matching future-feedback joint baseline too.
  Report this as a different supervision experiment from command imitation.
  Never copy the current observed pose into the current action and call it a
  future target; never shift across episode boundaries.

Do not use raw VR/PICO poses as robot TCP labels without the calibrated
retargeting transform. Check quaternion norms and reject invalid quaternions.
Convert each `[x,y,z,qx,qy,qz,qw]` to `[x,y,z,rx,ry,rz]` using normalized
`scipy.spatial.transform.Rotation.from_quat(q).as_rotvec()`. Positions must be
in metres and rotation vectors in radians. The template uses `XYZ_ROTVEC`,
which the upstream EEF processor supports; do not pass seven quaternion
coordinates under that format.

Use the following 40-value layout for both state and action:

| Key | Slice | Contents |
| --- | --- | --- |
| `body` | `[0:5]` | Body feedback/targets |
| `head` | `[5:8]` | Head feedback/targets |
| `left_eef` | `[8:14]` | Left XYZ + rotation vector |
| `right_eef` | `[14:20]` | Right XYZ + rotation vector |
| `left_hand` | `[20:30]` | Left hand feedback/valid targets |
| `right_hand` | `[30:40]` | Right hand feedback/valid targets |

Use shape `[40]` in metadata. Two 6D poses replace the fourteen arm joints of
the 42-value joint layout. If choosing `XYZ_ROT6D` instead, two 9D poses give
46 values and require new slices and matching formats.

Use [eef_config.py](eef_config.py). Each EEF is `EEF`, `XYZ_ROTVEC`, and
`RELATIVE`, with the same-named current pose as reference. The upstream
implementation uses `inverse(T_current) @ T_target`, including rotation of the
translation into the reference frame; simple quaternion or Euler subtraction
is incorrect. Store absolute poses and let GR00T do the conversion. Its policy
postprocessing reconstructs absolute actions; do not apply the delta twice.

This minimal EEF state omits arm joints. For redundant-arm disambiguation,
consider appending the 14 measured arm joints to the EEF **state only**, updating
the state metadata and config to 54 dimensions. The IK controller should use
current measured joints as its seed regardless. Body/head predictions must be
coordinated with the arm IK solution. Neither template fixes torso motion or
assumes a stationary torso without checking the demonstrations.

## GR00T metadata and training

Create `meta/modality.json` in each prepared root. Its `state` and `action`
objects map each table key to `{"start": start, "end": end}`. Add:

```json
{
  "video": {
    "head": {"original_key": "observation.images.head_stereo_left"},
    "left_wrist": {"original_key": "observation.images.hand_left_color"},
    "right_wrist": {"original_key": "observation.images.hand_right_color"}
  },
  "annotation": {
    "human.task_description": {"original_key": "annotation.human.task_description"}
  }
}
```

Merge those fields with the state/action objects; this fragment alone is not
a complete modality file. All config keys, slices, and feature dimensions must
agree. Do not reuse the source dataset's normalization statistics after changing
the arrays. Both templates register `NEW_EMBODIMENT`; load only one per process.

Install the main environment following the [upstream README](../../README.md#installation)
and the instructions for your GPU platform. From the repository root, with that
environment activated, run the following **after preparing the data and completing its checks**. Change `VARIANT=joints` to `VARIANT=eef` for the second run:

```bash
VARIANT=joints
python gr00t/data/stats.py \
  --dataset-path /data/g2-prepared/$VARIANT/train \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/AGIBOT_G2/${VARIANT}_config.py

python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /data/g2-prepared/$VARIANT/train \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/AGIBOT_G2/${VARIANT}_config.py \
  --output-dir /data/g2-runs/$VARIANT \
  --num-gpus 1 --global-batch-size 8 \
  --max-steps 2000 --save-steps 500 --save-total-limit 4 \
  --dataloader-num-workers 4
```

These are starting hyperparameters, not a measured optimum or a GPU-memory
guarantee. The default freezes the language/vision backbones and tunes the
projector/action model. First verify a small overfit run on one or two training
episodes, then run the full training split. Inspect camera crops to ensure the
bag, bin, and hands remain visible. Log the dataset and upstream revisions,
adapter revision, unit/frame conventions, split, horizon, seeds, and config.

## Evaluation and execution

Run upstream open-loop evaluation on a separate validation root:

```bash
python gr00t/eval/open_loop_eval.py \
  --dataset-path /data/g2-prepared/$VARIANT/validation \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path /data/g2-runs/$VARIANT/checkpoint-2000 \
  --traj-ids 0 --execution-horizon 16 --steps 400 \
  --save-plot-path /data/g2-runs/$VARIANT/validation-plots
```

Extend the episode IDs to all validation episodes. Checkpoint metadata carries
the selected modality configuration. Do not fit validation/test normalization
independently; use training/checkpoint statistics. Automatic training-time
evaluation is not a substitute for a verified held-out split.

Report per-joint error in radians (linear joints separately), hand-target error
on valid labels, TCP position error in metres, and rotation geodesic error
`acos(clip((trace(R_pred.T @ R_gt)-1)/2, -1, 1))`. Compute FK of the joint model's
predictions to compare both models in the same TCP space. Do not compare a
single aggregate MSE across different physical units and representations.
Check temporally constant predictions, identity-action baselines, command
latency, end-of-episode padding, and normalization saturation.

Select checkpoints on validation; report the untouched test set once. For real
task comparison use matched initial conditions and count completed bag-to-bin
placements, drops, collisions, interventions, and execution time. No task-success
rate has been measured in this fork.

Joint outputs need named-joint remapping, unit conversion, and a rate-limited
position controller. EEF outputs additionally need calibrated Cartesian control
or IK, reachability checks, joint limits, and a consistent TCP frame. Both need
hand commands and coordinated body/head control. Start with replay/simulation
and supervised low-speed trials. Predict a chunk, execute a short prefix, and
replan; measure inference latency to choose the prefix rather than assuming
the model can run at the recorder's 60 Hz rate.

## Remaining implementation work

- Review the adapter audit on the final cleaned snapshot and choose episode splits.
- Confirm G2 units, pose-stream semantics, joint names, URDF, and TCP transforms.
- Generate the two prepared dataset views and train separate checkpoints.
- Integrate joint and Cartesian execution, and compare held-out and robot results.

References: [data preparation](../../getting_started/data_preparation.md),
[modality configuration](../../getting_started/data_config.md),
[custom-embodiment training](../../getting_started/finetune_new_embodiment.md),
[v3 converter](../../scripts/lerobot_conversion/convert_v3_to_v2.py), and
[pose transforms](../../gr00t/data/state_action/pose.py).
