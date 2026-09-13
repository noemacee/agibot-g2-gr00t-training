# SPDX-License-Identifier: Apache-2.0
"""Predict both arm poses plus body, head, and hand joints in one action chunk.

Outputs: two 6D poses + 28 joint targets = 40 values per timestep.
Requires the prepared layout documented in README.md.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


KEYS = ["body", "head", "left_eef", "right_eef", "left_hand", "right_hand"]
# Only the arms use EEF actions. Every other key is a predicted joint target.
RELATIVE_KEYS = ["left_eef", "right_eef"]

config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(delta_indices=[0], modality_keys=KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=KEYS,
        action_configs=[
            ActionConfig(
                rep=(
                    ActionRepresentation.RELATIVE
                    if key in RELATIVE_KEYS
                    else ActionRepresentation.ABSOLUTE
                ),
                type=ActionType.EEF if key in RELATIVE_KEYS else ActionType.NON_EEF,
                format=ActionFormat.XYZ_ROTVEC if key in RELATIVE_KEYS else ActionFormat.DEFAULT,
            )
            for key in KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
