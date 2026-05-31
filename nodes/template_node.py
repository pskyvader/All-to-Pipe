"""
All-to-Pipe template node.

Assigns and validates template strings for dynamic prompt parsing.
"""

from typing import Any
from ..alltopipe_types import (
    Pipe,
    # PositivePrompt,
    # NegativePrompt,
    Template,
    TemplateParser,
)

# from ..common.utils import deep_copy_pipe


class TemplateNode:
    """
    Assigns and parses template strings for dynamic prompt generation.

    Templates use {variable} syntax to reference prompt attributes from
    PositivePrompt and NegativePrompt classes.

    Example: "A {age} {body} person wearing {clothes} in a {background}"

    Supports validation and parsing of templates against available prompt data.
    """

    def __init__(self) -> None:
        """Initialize the template node."""
        pass

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
                "template_type": (["positive", "negative"],),
                "template_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A person wearing <color> <clothes>",
                    },
                ),
                "allow_missing": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES: tuple[str, ...] = ("PIPE", "STRING")
    RETURN_NAMES: tuple[str, ...] = ("pipe", "parsed_template")
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"

    def execute(
        self,
        template_type: str,
        template_text: str,
        allow_missing: bool,
        pipe: Pipe | None = None,
    ) -> tuple[Pipe, str]:
        """
        Execute the node and assign template to pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            template_type: "positive" or "negative"
            template_text: Template string with <variable> placeholders
            Example: "A <age> <body> wearing <clothes>"

        Returns:
            Tuple containing the modified Pipe instance

        Raises:
            ValueError: If template_type is invalid
        """
        # new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()
        new_pipe: Pipe = pipe.clone() if pipe is not None else Pipe()
        # Validate template syntax (find placeholders)
        placeholders = TemplateParser.find_placeholders(template_text)
        template = Template(template_type, placeholders, template_text, allow_missing)

        if template_type == "positive":
            # Store template in positive prompt
            if new_pipe.positive_prompt is not None:
                template.parsed_template = TemplateParser.parse_template(
                    template_text,
                    new_pipe.positive_prompt,
                    allow_missing,
                )
            new_pipe.positive_template = template

        else:  # negative
            # Store template in negative prompt
            if new_pipe.negative_prompt is not None:
                template.parsed_template = TemplateParser.parse_template(
                    template_text,
                    new_pipe.negative_prompt,
                    allow_missing,
                )

            new_pipe.negative_template = template

        return (new_pipe, template.parsed_template if template.parsed_template else "")
