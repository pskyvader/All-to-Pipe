"""
All-to-Pipe prompt template parser.

System for parsing text templates with variable substitution from prompt objects.
"""

import re
from typing import Any
from .prompts import PositivePrompt, NegativePrompt


class Template:
    def __init__(
        self,
        template_type: str,
        placeholders: list[str],
        text: str,
        allow_missing: bool,
    ) -> None:

        if template_type not in ["positive", "negative"]:
            raise ValueError("template_type must be 'positive' or 'negative'")

        # If template is empty, skip
        if not text.strip():
            raise ValueError("template_text cannot be empty")

        self.type: str = template_type
        self.placeholders: list[str] = placeholders
        self.text: str = text
        self.allow_missing: bool = allow_missing
        self.parsed_template: str | None = None


class TemplateParser:
    """
    Parser for prompt templates with variable substitution.

    Supports templates with <variable> placeholders that are substituted
    from PositivePrompt and NegativePrompt attributes.

    Uses angle brackets <> to avoid conflicts with comfyui-dynamic-prompts module
    which uses {}, __term__, $, (), | and other symbols.
    """

    # Regex pattern for finding placeholders with <variable> syntax
    PLACEHOLDER_PATTERN = re.compile(r"<([^>]+)>")

    @staticmethod
    def find_placeholders(template: str) -> list[str]:
        """
        Find all placeholders in a template string.

        Args:
            template: Template string with <variable> placeholders

        Returns:
            List of variable names found in the template

        Raises:
            ValueError: If template is None or not a string

        Example:
            >>> TemplateParser.find_placeholders("A <age> person")
            ['age']
        """
        matches = TemplateParser.PLACEHOLDER_PATTERN.findall(template)
        return matches

    @staticmethod
    def parse_template(
        template: str,
        prompt_map: PositivePrompt | NegativePrompt,
        allow_missing: bool = False,
    ) -> str:
        """
        Parse a template string with variable substitution from prompt objects.

        Replaces <variable> placeholders with values from prompt attributes.

        Args:
            template: Template string with <variable> placeholders
            prompt_map: PositivePrompt or NegativePrompt instance with substitution values
            allow_missing: If True, replace missing values with empty string. If False, use placeholder.

        Returns:
            Parsed template string with all variables substituted

        Example:
            >>> pos = PositivePrompt()
            >>> pos.age = "young"
            >>> TemplateParser.parse_template("A <age> person", pos)
            'A young person'
        """

        result: str = template
        placeholders = TemplateParser.find_placeholders(template)

        for placeholder in placeholders:
            # Try to get from positive prompt first
            value: str | None = None

            if hasattr(prompt_map, placeholder):
                value = getattr(prompt_map, placeholder)

            if value is None:
                if allow_missing:
                    # Use default value for missing variables
                    value = ""
                else:
                    value = "<MISSING " + placeholder + " " + ">"

            # Replace placeholder with value (using <> syntax)
            result = result.replace(f"<{placeholder}>", str(value))

        result = " ".join(result.split())
        result = ",".join(result.split(", ,"))
        result = ",".join(result.split(","))
        result = re.sub(r",+", ",", result)  # ",," -> ","
        result = re.sub(r"\s*,\s*", ", ", result)  # "word ,word" -> "word, word"
        result = result.strip().strip(",")

        return result

    # @staticmethod
    # def get_required_variables(template: str) -> List[str]:
    #     """
    #     Get list of required variables from a template.

    #     Args:
    #         template: Template string with placeholders

    #     Returns:
    #         List of required variable names (no duplicates)

    #     Raises:
    #         ValueError: If template is not a string
    #     """

    #     placeholders = TemplateParser.find_placeholders(template)
    #     # Remove duplicates while preserving order
    #     seen: Set[str] = set()
    #     # unique = []
    #     for p in placeholders:
    #         if p not in seen:
    #             seen.add(p)
    #             # unique.append(p)
    #     return list(seen)
