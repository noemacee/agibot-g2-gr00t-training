# SPDX-License-Identifier: Apache-2.0
"""Predict all 42 recorded body, head, arm, and hand joints in one action chunk.

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


KEYS = ["body", "head", "left_arm", "right_arm", "left_hand", "right_hand"]
RELATIVE_KEYS = ["left_arm", "right_arm"]

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
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
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
