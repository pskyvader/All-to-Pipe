"""
All-to-Pipe utils module.

Utility functions for common operations across nodes.
"""

import copy
from ..alltopipe_types import (
    Pipe,
    Model,
    LoraSpec,
    Parameters,
    ImageConfig,
    PositivePrompt,
    NegativePrompt,
)

# TODO: question, is it really necessary? isnt comfy supposed to already manage that the nodes are immutable?


def deep_copy_pipe(pipe: Pipe | None) -> Pipe:
    """
    Create a safe deep copy of a Pipe instance.

    This ensures that modifications to the copy do not affect the original.

    Args:
        pipe: The Pipe instance to copy

    Returns:
        A new Pipe instance with deep-copied data
    """
    if pipe is None:
        raise ValueError("Cannot copy None pipe")

    # Deep copy model
    new_model: Model | None = None
    if pipe.model is not None:
        new_model = Model(
            name=pipe.model.name,
            subfolder=pipe.model.subfolder,
            clip_skip=pipe.model.clip_skip,
        )

    # Deep copy loras list
    new_loras: list[LoraSpec] = []
    if pipe.loras:
        for lora in pipe.loras:
            new_lora: LoraSpec = LoraSpec(
                name=lora.name,
                subfolder=lora.subfolder,
                weight=lora.weight,
                clip_weight=lora.clip_weight,
            )
            new_loras.append(new_lora)

    # Deep copy parameters
    new_parameters: Optional[Parameters] = None
    if pipe.parameters is not None:
        new_parameters = Parameters(
            steps=pipe.parameters.steps,
            cfg=pipe.parameters.cfg,
            sampler=pipe.parameters.sampler,
            scheduler=pipe.parameters.scheduler,
            seed=pipe.parameters.seed,
            denoise=pipe.parameters.denoise,
        )

    # Deep copy image config
    new_image_config: Optional[ImageConfig] = None
    if pipe.image_config is not None:
        new_image_config = ImageConfig(
            width=pipe.image_config.width,
            height=pipe.image_config.height,
            batch_size=pipe.image_config.batch_size,
            noise=pipe.image_config.noise,
            color_code=pipe.image_config.color_code,
            image=pipe.image_config.image,
        )

    # Deep copy prompts
    new_positive_prompt: PositivePrompt = PositivePrompt()
    new_negative_prompt: NegativePrompt = NegativePrompt()

    # Copy all attributes from original prompts to new prompts
    if pipe.positive_prompt is not None:
        for key, value in pipe.positive_prompt.__dict__.items():
            setattr(new_positive_prompt, key, copy.deepcopy(value))

    if pipe.negative_prompt is not None:
        for key, value in pipe.negative_prompt.__dict__.items():
            setattr(new_negative_prompt, key, copy.deepcopy(value))

    # Create new pipe with deep-copied data
    new_pipe: Pipe = Pipe(
        model=new_model,
        loras=new_loras,
        parameters=new_parameters,
        image_config=new_image_config,
        positive_prompt=new_positive_prompt,
        negative_prompt=new_negative_prompt,
    )

    # Deep copy companion data
    if pipe.companion_model_data is not None:
        new_pipe.companion_model_data = copy.deepcopy(pipe.companion_model_data)

    if pipe.companion_lora_data is not None:
        new_pipe.companion_lora_data = copy.deepcopy(pipe.companion_lora_data)

    return new_pipe
