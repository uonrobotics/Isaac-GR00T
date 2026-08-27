from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


sign_nav_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
        # modality_keys=["rgb_ego_view","segmented_ego_view"]
    ),

    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "speed",
        ],
    ),

    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "vel_cmd",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",
        ],
    ),
}


register_modality_config(
    sign_nav_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)