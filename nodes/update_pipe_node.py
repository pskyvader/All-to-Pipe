"""
All-to-Pipe pipe node.

Creates an empty Pipe object as the entry point of the pipeline.
"""

from typing import Any
from ..alltopipe_types import Pipe
from ..alltopipe_types import ImageConfig


class UpdatePipeNode:
    elements_to_update: dict[str, dict[str, Any]] = {
        "image": {
            "class": ImageConfig,
            "attr_name": "image",  # The attribute inside ImageConfig
            "pipe_key": "image_config",  # The attribute inside Pipe
            "required": True,
        }
    }

    def __init__(self) -> None:
        """Initialize the pipe node."""
        pass

    @staticmethod
    def execute(key: str, value: Any, pipe: Pipe | None = None) -> tuple[Pipe]:
        new_pipe = pipe.clone() if pipe is not None else Pipe()

        element = UpdatePipeNode.elements_to_update[key]
        pipe_attr = element["pipe_key"]

        updated_instance = getattr(new_pipe, pipe_attr, None)
        if updated_instance is None:
            if element["required"]:
                raise ValueError(
                    f"Required attribute {element['attr_name']} not found in pipe"
                )

            updated_instance = element["class"]()

        setattr(updated_instance, element["attr_name"], value)

        if element["attr_name"] == "image":
            if new_pipe.model is not None and new_pipe.model.cached_model is not None:
                _, _, vae = new_pipe.model.cached_model
                updated_instance.latent = {
                    "samples": vae.encode(updated_instance.image[:, :, :, :3])
                }
            else:
                print("No model found for encoding image")

        setattr(new_pipe, pipe_attr, updated_instance)

        return (new_pipe,)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {"key": (list(cls.elements_to_update.keys()),), "value": "*"},
        }

    RETURN_TYPES: tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
