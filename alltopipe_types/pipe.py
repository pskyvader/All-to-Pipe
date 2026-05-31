"""
Pipe type for All-to-Pipe.

Central container that holds all pipeline data.
"""

from typing import Any, Self
from .model import Model
from .lora import LoraSpec
from .parameters import Parameters
from .image_config import ImageConfig
from .prompts import PositivePrompt, NegativePrompt
from .template import Template


class Pipe:
    """
    Central container passed between nodes.
    Holds unresolved (non-encoded) data only.
    Nothing is loaded or encoded here.
    """

    def __init__(
        self,
        model: Model | None = None,
        loras: list[LoraSpec] | None = None,
        parameters: Parameters | None = None,
        image_config: ImageConfig | None = None,
        positive_template: Template | None = None,
        negative_template: Template | None = None,
        positive_prompt: PositivePrompt | None = None,
        negative_prompt: NegativePrompt | None = None,
        companion_model_data: dict[str, Any] | None = None,
        companion_lora_data: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Initialize Pipe container.

        Args:
            model: Model reference
            loras: List of LoRA specifications
            parameters: Generation parameters
            image_config: Image configuration settings
            positive_prompt: Positive prompt container
            negative_prompt: Negative prompt container
        """
        self.model: Model | None = model
        self.loras: list[LoraSpec] = loras if loras is not None else []
        self.parameters: Parameters | None = parameters
        self.image_config: ImageConfig | None = image_config
        self.positive_template: Template | None = positive_template
        self.negative_template: Template | None = negative_template

        self.positive_prompt: PositivePrompt | None = (
            positive_prompt if positive_prompt is not None else PositivePrompt()
        )
        self.negative_prompt: NegativePrompt | None = (
            negative_prompt if negative_prompt is not None else NegativePrompt()
        )
        # Companion file data from models/LoRAs
        self.companion_model_data = companion_model_data
        self.companion_lora_data = companion_lora_data

    def clone(self) -> Self:
        """Returns a deep copy of the current Pipe instance."""
        return type(self)(
            model=self.model,
            loras=list(self.loras),
            parameters=self.parameters,
            image_config=self.image_config,
            positive_template=self.positive_template,
            negative_template=self.negative_template,
            positive_prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            companion_model_data=(
                dict(self.companion_model_data)
                if self.companion_model_data is not None
                else None
            ),
            companion_lora_data=(
                [dict(d) for d in self.companion_lora_data]
                if self.companion_lora_data is not None
                else None
            ),
        )

    def derive(self, **overrides: Any) -> Self:
        """
        A clever way to clone AND update specific values at once.
        Example: new_pipe = pipe.derive(positive_prompt="A sunny day")
        """
        new_instance = self.clone()
        for key, value in overrides.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
        return new_instance
