"""
All-to-Pipe image config node.

Sets image dimensions, batch size, and noise parameters.
"""

from typing import Any
from ..alltopipe_types import Pipe, ImageConfig, ImageConfigProcessor
from ..common.utils import deep_copy_pipe


class ImageConfigNode:
    """
    Sets image configuration in the Pipe.

    Specifies dimensions, batch size, and noise parameters for generation.
    """

    def __init__(self) -> None:
        """Initialize the image config node."""
        pass

    @staticmethod
    def execute(
        pipe: Pipe | None = None,
        width: int = 512,
        height: int = 512,
        batch_size: int = 1,
        noise: float = 1.0,
        color_code: str = "",
    ) -> tuple[Pipe]:
        """
        Execute the node and set image configuration in the pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            width: Image width in pixels
            height: Image height in pixels
            batch_size: Number of images in batch
            noise: Noise level (0.0-1.0 percentage)
            color_code: Hex color code (e.g., "#FF0000") or empty for random

        Returns:
            Tuple containing the modified Pipe instance
        """
        # Deep copy pipe to avoid modifying the original
        # new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()
        new_pipe: Pipe = pipe.clone() if pipe is not None else Pipe()

        # Create and attach image config
        color: str | None = color_code if color_code.strip() else None
        image_config: ImageConfig = ImageConfig(
            width=width,
            height=height,
            batch_size=batch_size,
            noise=noise,
            color_code=color,
        )
        # seed = new_pipe.parameters.seed if new_pipe.parameters else None

        # image_config.image = ImageConfigProcessor.create_noisy_image(image_config, seed)
        # if new_pipe.model is not None and new_pipe.model.cached_model is not None:
        #     _, _, vae = new_pipe.model.cached_model
        #     image_config.latent = {
        #         "samples": vae.encode(image_config.image[:, :, :, :3])
        #     }

        new_pipe.image_config = image_config

        return (new_pipe,)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
        """
        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "noise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "color_code": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES: tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
