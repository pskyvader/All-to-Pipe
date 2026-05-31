"""
All-to-Pipe export node.

Resolves Pipe into sampler-ready objects.
"""

from typing import Any
from ..alltopipe_types import (
    Pipe,
    Model,
    ModelProcessor,
    LoraProcessor,
    ImageConfigProcessor,
    PromptProcessor,
    Parameters,
    ImageConfig,
    PositivePrompt,
    NegativePrompt,
    Template,
    TemplateParser,
)


from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES
import torch

# import gc
# import comfy.model_management


class ExportNode:
    """
    Resolves Pipe into sampler-ready objects with KSampler-compatible outputs.

    - Loads model
    - Applies all LoRAs
    - Encodes prompts with template parsing support
    - Exports all parameters individually for direct KSampler connection
    - Creates empty LATENT image from image config dimensions
    - Returns: model, positive conditioning, negative conditioning, 6 sampling parameters, and latent

    All outputs connect directly to standard ComfyUI KSampler node:
    - model → KSampler.model
    - positive → KSampler.positive
    - negative → KSampler.negative
    - seed → KSampler.seed
    - steps → KSampler.steps
    - cfg → KSampler.cfg
    - sampler → KSampler.sampler_name
    - scheduler → KSampler.scheduler
    - denoise → KSampler.denoise
    - latent_image → KSampler.latent_image
    """

    def __init__(self) -> None:
        """Initialize the export node."""
        pass

    @staticmethod
    def execute(pipe: Pipe) -> tuple[Any, ...]:
        """
        Execute the node and resolve pipe into sampler-ready objects.

        All parameters are exported individually for direct connection to KSampler nodes.
        Creates an empty LATENT image from image configuration dimensions.

        Args:
            pipe: The input Pipe instance (must be fully configured)

        Returns:
            Tuple containing 10 individual values:
                - model: Loaded MODEL object (or None)
                - positive: Encoded CONDITIONING from positive prompt
                - negative: Encoded CONDITIONING from negative prompt
                - seed: Random seed (INT)
                - steps: Number of sampling steps (INT)
                - cfg: Classifier-free guidance scale (FLOAT)
                - sampler: Sampler algorithm name (STRING)
                - scheduler: Scheduler algorithm name (STRING)
                - denoise: Denoise strength (FLOAT, range 0.0-1.0)
                - latent_image: Empty LATENT image tensor (created from image_config dimensions)

        Raises:
            ValueError: If pipe is invalid or incomplete
        """
        if not isinstance(pipe.model, Model):
            raise ValueError("Pipe.model must be a Model instance")
        if not isinstance(pipe.parameters, Parameters):
            raise ValueError("Pipe.parameters must be a Parameters instance")
        if not isinstance(pipe.image_config, ImageConfig):
            raise ValueError("Pipe.image_config must be an ImageConfig instance")
        if not isinstance(pipe.positive_prompt, PositivePrompt):
            raise ValueError("Pipe.positive_prompt must be a PositivePrompt instance")
        if not isinstance(pipe.negative_prompt, NegativePrompt):
            raise ValueError("Pipe.negative_prompt must be a NegativePrompt instance")
        if not isinstance(pipe.positive_template, Template):
            raise ValueError("Pipe.positive_template must exist")
        if not isinstance(pipe.negative_template, Template):
            raise ValueError("Pipe.negative_template must exist")

        (model, clip, vae) = ModelProcessor.load_model(pipe.model)

        # Apply LoRAs if model was loaded and LoRAs are specified
        if pipe.loras:
            model, clip = LoraProcessor.apply_lora(model, clip, pipe.loras)

        if pipe.positive_template.parsed_template is None:
            pipe.positive_template.parsed_template = TemplateParser.parse_template(
                pipe.positive_template.text,
                pipe.positive_prompt,
                pipe.positive_template.allow_missing,
            )
        positive_conditioning = PromptProcessor.encode_prompt(
            clip,
            pipe.positive_template.parsed_template,
            pipe.image_config.width,
            pipe.image_config.height,
            pipe.image_config.width,
            pipe.image_config.height,
            crop_w=0,
            crop_h=0,
        )

        if pipe.negative_template.parsed_template is None:
            pipe.negative_template.parsed_template = TemplateParser.parse_template(
                pipe.negative_template.text,
                pipe.negative_prompt,
                pipe.negative_template.allow_missing,
            )
        negative_conditioning = PromptProcessor.encode_prompt(
            clip,
            pipe.negative_template.parsed_template,
            pipe.image_config.width,
            pipe.image_config.height,
            pipe.image_config.width,
            pipe.image_config.height,
            crop_w=0,
            crop_h=0,
        )

        # Extract all parameters for direct KSampler connection
        seed: int = pipe.parameters.seed
        steps: int = pipe.parameters.steps
        cfg: float = pipe.parameters.cfg
        sampler_name: str = pipe.parameters.sampler
        scheduler: str = pipe.parameters.scheduler
        denoise: float = pipe.parameters.denoise

        if not isinstance(pipe.image_config.image, torch.Tensor):
            pipe.image_config.image = ImageConfigProcessor.create_noisy_image(
                pipe.image_config, seed
            )
        image = pipe.image_config.image

        if not isinstance(pipe.image_config.latent, dict):
            pipe.image_config.latent = {"samples": vae.encode(image[:, :, :, :3])}

        latent_image = pipe.image_config.latent

        # since this node does all at the same time, it's important to clean up
        # the pipe
        # del pipe
        # gc.collect()
        # comfy.model_management.soft_empty_cache()

        # Return all parameters individually for direct KSampler connection
        # This allows connecting each output directly to KSampler inputs
        return (
            pipe,
            model,
            vae,
            clip,
            positive_conditioning,
            negative_conditioning,
            image,
            latent_image,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
        )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining node inputs
        """
        return {
            "required": {
                "pipe": ("PIPE",),
            }
        }

    RETURN_TYPES: tuple[Any, ...] = (
        "PIPE",
        "MODEL",
        "VAE",
        "CLIP",
        "CONDITIONING",
        "CONDITIONING",
        "IMAGE",
        "LATENT",
        "INT",
        "INT",
        "FLOAT",
        SAMPLER_NAMES,
        SCHEDULER_NAMES,
        "FLOAT",
    )
    RETURN_NAMES: tuple[str, ...] = (
        "Pipe",
        "model",
        "vae",
        "clip",
        "positive",
        "negative",
        "image",
        "latent_image",
        "seed",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "denoise",
    )
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
