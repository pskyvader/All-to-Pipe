"""
All-to-Pipe validators module.

Cross-node validation functions.
"""

from ..alltopipe_types import (
    Pipe,
    Model,
    Parameters,
    ImageConfig,
    PositivePrompt,
    NegativePrompt,
)


def validate_pipe(pipe: Pipe) -> None:
    """
    Validate that a Pipe contains all required fields.

    This is a cross-node validator used by export nodes.

    Raises:
        ValueError: If required fields are missing or invalid

    Args:
        pipe: The Pipe instance to validate
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
    if not pipe.positive_prompt.template:
        raise ValueError("Pipe.positive_prompt.template must be a string")
    if not pipe.negative_prompt.template:
        raise ValueError("Pipe.negative_prompt.template must be a string")
