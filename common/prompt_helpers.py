"""
All-to-Pipe prompt helpers module.

Helper functions for working with prompt data structures.
"""

from ..alltopipe_types import PositivePrompt, NegativePrompt

# TODO: check template node to know how to parse the templates and classes correctly


def prompt_to_string(prompt: PositivePrompt | NegativePrompt) -> dict[str, str]:
    """
    Convert prompt object attributes into plain strings.

    This function extracts all string attributes from a prompt object
    and returns them as a dictionary.

    Args:
        prompt: Either a PositivePrompt or NegativePrompt instance

    Returns:
        Dictionary mapping attribute names to their string values

    Raises:
        ValueError: If prompt is None
    """

    prompt_dict: dict[str, str] = {}

    # Extract all attributes from the prompt object
    for key, value in prompt.__dict__.items():
        if isinstance(value, str):
            prompt_dict[key] = value
        elif value is not None:
            # Convert non-string values to strings
            prompt_dict[key] = str(value)

    return prompt_dict


def merge_prompts(prompts: dict[str, str], separator: str = ", ") -> str:
    """
    Merge multiple prompt attributes into a single string.

    Args:
        prompts: Dictionary of prompt attributes
        separator: String to use when joining prompt parts

    Returns:
        Merged prompt string
    """
    if not prompts:
        return ""

    prompt_parts: list[str] = [v for v in prompts.values() if v]
    return separator.join(prompt_parts)
