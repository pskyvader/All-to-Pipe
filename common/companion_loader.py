"""
Companion file loader for All-to-Pipe.

Loads and manages JSON companion files for models and LoRAs.
"""

from typing import Any
import json
import os
import random
import logging
import re
from dataclasses import dataclass, field

from ..alltopipe_types import Model, Parameters, ImageConfig

# Setup logging for companion file warnings
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


@dataclass
class CompanionFile:
    """
    Represents a companion JSON file for a model or LoRA.

    Provides metadata, limits, and prompt suggestions.
    """

    sampler: list[str] = field(default_factory=list)
    """List of samplers. Single value means select one from list."""
    scheduler: list[str] = field(default_factory=list)

    steps: list[int] = field(default_factory=list)
    """
    Steps configuration:
    - len=1: use that specific value
    - len=2: range [min, max], select random in range
    - len>2: choose one from list
    """

    resolution: list[int] = field(default_factory=list)
    """List of valid [width, height] pairs."""

    cfg: list[float] = field(default_factory=list)
    """Cfg configuration (same as steps)."""

    negative_prompt: list[str] = field(default_factory=list)
    """List of negative prompts to randomly select from."""

    positive_prompt: list[str] = field(default_factory=list)
    """List of positive prompts to randomly select from."""

    raw_data: dict[str, Any] = field(default_factory=dict)
    """Raw dictionary data from JSON file."""

    clip_skip: list[int] = field(default_factory=list)
    """List of clip skip values to randomly select from."""


class CompanionLoader:
    """
    Loader for companion JSON files.

    Handles discovering, loading, and parsing companion files
    for models and LoRAs.
    """

    @staticmethod
    def load_model_companion(
        model_name: str,
        subfolder: str,
        base_path: str = "models/checkpoints",
    ) -> CompanionFile | None:
        """
        Load companion JSON file for a model.

        Looks for {model_name}.json alongside the model file.

        Args:
            model_name: Model filename without extension
            subfolder: Subfolder within models/checkpoints
            base_path: Base path to models directory

        Returns:
            CompanionFile instance if found, None otherwise.
            Logs a warning if file is not found.
        """
        result = CompanionLoader._load_companion_file(model_name, subfolder, base_path)
        if result is None:
            logger.warning(
                f"No companion file found for model '{model_name}' in {subfolder or 'root'}. "
                f"Expected: {os.path.join(base_path, subfolder or '', f'{model_name}.json')}"
            )
        return result

    @staticmethod
    def load_lora_companion(
        lora_name: str,
        subfolder: str,
        base_path: str = "models/loras",
    ) -> CompanionFile | None:
        """
        Load companion JSON file for a LoRA.

        Looks for {lora_name}.json alongside the LoRA file.
        Automatically removes file extensions (e.g., .safetensors) before searching.

        Args:
            lora_name: LoRA filename (with or without extension)
            subfolder: Subfolder within models/loras
            base_path: Base path to loras directory

        Returns:
            CompanionFile instance if found, None otherwise.
            Logs a warning if file is not found.
        """
        result = CompanionLoader._load_companion_file(lora_name, subfolder, base_path)
        if result is None:
            logger.warning(
                f"No companion file found for LoRA '{lora_name}' in {subfolder or 'root'}. "
                f"Expected: {os.path.join(base_path, subfolder or '', f'{lora_name}.json')}"
            )
        return result

    @staticmethod
    def _load_companion_file(
        name: str,
        subfolder: str,
        base_path: str,
    ) -> CompanionFile | None:
        """
        Internal method to load companion JSON file.

        Args:
            name: Base name (without extension)
            subfolder: Subfolder path
            base_path: Base directory path

        Returns:
            CompanionFile instance if found, None otherwise
        """
        if not name:
            return None

        base = name.rsplit(".", 1)[0]
        json_name = base + ".json"

        # Build path: base_path/subfolder/name.json
        if subfolder:
            json_path: str = os.path.join(base_path, subfolder, json_name)
        else:
            json_path = os.path.join(base_path, json_name)

        # Check if file exists
        if not os.path.isfile(json_path):
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)

            # Parse JSON into CompanionFile
            return CompanionLoader._parse_companion_data(data)

        except (json.JSONDecodeError, IOError, OSError):
            # Log error silently, return None
            return None

    @staticmethod
    def apply_companion_to_parameters(
        companion: CompanionFile,
        parameters: Parameters,
    ) -> Parameters:
        """
        Apply companion file values to generation parameters intelligently.

        Args:
            companion: CompanionFile instance (or None)
            steps: Current steps value
            cfg: Current cfg value
            sampler: Current sampler
            scheduler: Current scheduler
            seed: Current seed

        Returns:
            Tuple of (new_steps, new_cfg, new_sampler, new_scheduler, new_seed)
        """

        steps, cfg, sampler, scheduler, seed = (
            parameters.steps,
            parameters.cfg,
            parameters.sampler,
            parameters.scheduler,
            parameters.seed,
        )

        # Apply sampler (single value or choice from list)
        if companion.sampler:
            sampler = CompanionLoader._apply_choice_value(
                companion.sampler, sampler, "sampler"
            )

        # Apply steps (numeric type)
        if companion.steps:
            steps = int(
                CompanionLoader._apply_numeric_value(companion.steps, steps, "steps")
            )

        # Apply cfg (numeric type)
        if companion.cfg:
            cfg = CompanionLoader._apply_numeric_value(companion.cfg, cfg, "cfg")

        if companion.scheduler:
            scheduler = CompanionLoader._apply_choice_value(
                companion.scheduler, scheduler, "scheduler"
            )

        (
            parameters.steps,
            parameters.cfg,
            parameters.sampler,
            parameters.scheduler,
            parameters.seed,
        ) = (steps, cfg, sampler, scheduler, seed)

        return parameters

    @staticmethod
    def apply_companion_to_image_config(
        companion: CompanionFile,
        image_config: ImageConfig | None,
    ) -> ImageConfig:
        """
        Apply companion file resolution to image config intelligently.

        Handles resolution as pairs [[w,h], [w,h], ...] or string formats.

        Args:
            companion: CompanionFile instance (or None)
            width: Current width
            height: Current height
            batch_size: Current batch_size

        Returns:
            Tuple of (new_width, new_height, new_batch_size)
        """
        # Process resolution pairs
        resolutions: list[tuple[int, int]] = []
        for res in companion.resolution:
            if isinstance(res, list) and len(res) == 2:
                resolutions.append((int(res[0]), int(res[1])))
            elif isinstance(res, str):
                # Parse string formats like "1024x768", "1024,768", "1024 x 768"
                parsed = CompanionLoader._parse_resolution_string(res)
                if parsed:
                    resolutions.append(parsed)
            elif isinstance(res, dict) and "width" in res and "height" in res:
                # Dict format {width: ..., height: ...}
                resolutions.append((int(res["width"]), int(res["height"])))

        if not resolutions:
            raise ValueError("No valid resolution found in companion")

        if not image_config:
            image_config = ImageConfig(-1, -1, 1)

        current_res = (
            image_config.width,
            image_config.height,
        )
        if current_res not in resolutions:
            # Current resolution not in list, pick a random one
            new_w, new_h = random.choice(resolutions)
            logger.info(
                f"Current resolution {current_res} not in companion list, "
                f"selected random: {new_w}x{new_h}"
            )
            image_config.width, image_config.height = new_w, new_h

        return image_config

    @staticmethod
    def apply_companion_to_prompts(
        companion: CompanionFile,
        positive_prompt: str | None,
        negative_prompt: str | None,
    ) -> tuple[str, str]:
        """
        Apply companion prompt suggestions to prompts intelligently.

        Can add from lists or comma-separated strings, with random selection.

        Args:
            companion: CompanionFile instance (or None)
            positive_prompt: Current positive prompt
            negative_prompt: Current negative prompt

        Returns:
            Tuple of (new_positive_prompt, new_negative_prompt)
        """
        if not positive_prompt:
            positive_prompt = ""
        if not negative_prompt:
            negative_prompt = ""
        # Apply positive prompt suggestions
        if companion.positive_prompt:
            positive_prompt = CompanionLoader.apply_text_suggestions(
                companion.positive_prompt, positive_prompt, "positive_prompt"
            )

        # Apply negative prompt suggestions
        if companion.negative_prompt:
            negative_prompt = CompanionLoader.apply_text_suggestions(
                companion.negative_prompt, negative_prompt, "negative_prompt"
            )

        return positive_prompt, negative_prompt

    @staticmethod
    def apply_companion_to_model(
        companion: CompanionFile,
        model: Model,
    ) -> Model:
        """
        Apply companion to model, if applicable."""

        clip_skip = -abs(model.clip_skip)
        if companion.clip_skip:
            clip_skip = int(
                CompanionLoader._apply_numeric_value(
                    companion.clip_skip, clip_skip, "clip_skip"
                )
            )
        model.clip_skip = clip_skip
        return model

    # ======================== Helper Methods ========================

    @staticmethod
    def _apply_numeric_value(
        values: list[int] | list[float],
        current: int | float,
        param_name: str,
    ) -> int | float:
        """
        Apply numeric values based on type:
        - Single value: use it directly
        - Two values [min, max]: check current is within range, else random in range
        - Multiple values: check current matches one, else choose random
        """
        current_type = type(current)
        if not values:
            return current_type(current)

        if isinstance(values, (int, float)):
            return current_type(values)

        if len(values) == 1:
            # Single value, use it directly
            val = values[0]
            return current_type(val)

        if len(values) == 2:
            # Range [min, max]
            min_val, max_val = values[0], values[1]
            if min_val > max_val:
                min_val, max_val = values[1], values[0]

            if min_val <= current <= max_val:
                # Current value within range, keep it
                return current_type(current)
            else:
                # Current value outside range, pick random in range
                if isinstance(current, int):
                    new_val = random.randint(int(min_val), int(max_val))
                else:
                    new_val = random.uniform(min_val, max_val)
                logger.info(
                    f"Value {current} for {param_name} outside range [{min_val}, {max_val}], "
                    f"selected random: {new_val}"
                )
                return current_type(new_val)

        # Multiple values, treat as choice list
        return current_type(
            CompanionLoader._apply_choice_value(values, current, param_name)
        )

    @staticmethod
    def _apply_choice_value(
        options: list[Any],
        current: Any,
        param_name: str,
    ) -> Any:
        """
        Apply choice values from a list:
        - If current matches one, keep it
        - Otherwise randomly choose from list
        """
        current_type = type(current)

        if not options:
            return current_type(current)

        if current in options:
            # Current value is valid, keep it
            return current_type(current)

        # Current not in options, pick random
        new_val = random.choice(options)
        logger.info(
            f"Value '{current}' for {param_name} not in companion list {options}, "
            f"selected random: '{new_val}'"
        )
        return current_type(new_val)

    @staticmethod
    def apply_text_suggestions(
        suggestions: list[str],
        current: str | None,
        param_name: str,
    ) -> str:
        """
        Apply text suggestions:
        - Parse suggestions (list or comma-separated strings)
        - Randomly add 0-50% of suggestions to current prompt
        """
        if not current:
            current = ""
        if not suggestions:
            return current
        # Flatten suggestions into list of terms
        all_terms: list[str] = []
        for suggestion in suggestions:
            terms = [t.strip() for t in suggestion.split(",") if t.strip()]
            all_terms.extend(terms)
        if not all_terms:
            return current
        subset_size = random.randint(1, len(all_terms))
        selected_terms = random.sample(all_terms, min(subset_size, len(all_terms)))
        current += ", ".join(selected_terms)
        logger.info(f"Added {subset_size} terms to {param_name}: {selected_terms}")
        return current

    @staticmethod
    def _parse_resolution_string(res_str: str) -> tuple[int, int] | None:
        """
        Parse resolution string formats like "1024x768", "1024,768", "1024 x 768".

        Returns:
            Tuple of (width, height) or None if parse fails
        """
        # Try patterns: WIDTHxHEIGHT, WIDTH,HEIGHT, WIDTH x HEIGHT
        patterns = [
            r"(\d+)\s*[x×]\s*(\d+)",  # 1024x768 or 1024 x 768
            r"(\d+)\s*,\s*(\d+)",  # 1024,768
            r"(\d+)\s+(\d+)",  # 1024 768
        ]

        for pattern in patterns:
            match = re.match(pattern, res_str, re.IGNORECASE)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                return (w, h)

        logger.warning(f"Could not parse resolution string: {res_str}")
        return None

    @staticmethod
    def _parse_companion_data(data: dict[str, Any]) -> CompanionFile:
        """
        Parse raw JSON data into CompanionFile.

        Args:
            data: Dictionary from JSON file

        Returns:
            Parsed CompanionFile instance
        """
        companion: CompanionFile = CompanionFile()
        companion.raw_data = data

        # Parse sampler
        if "sampler" in data:
            samplers: Any = data["sampler"]
            companion.sampler = samplers if isinstance(samplers, list) else [samplers]

        # Parse scheduler
        if "scheduler" in data:
            schedulers: Any = data["scheduler"]
            companion.scheduler = (
                schedulers if isinstance(schedulers, list) else [schedulers]
            )

        # Parse steps
        if "steps" in data:
            steps_data: Any = data["steps"]
            companion.steps = (
                steps_data if isinstance(steps_data, list) else [steps_data]
            )

        # Parse cfg
        if "cfg" in data:
            cfg_data: Any = data["cfg"]
            companion.cfg = cfg_data if isinstance(cfg_data, list) else [cfg_data]

        # Parse resolution
        if "resolution" in data:
            res_data: Any = data["resolution"]
            companion.resolution = (
                res_data if isinstance(res_data, list) else [res_data]
            )

        # Parse negative_prompt
        if "negative_prompt" in data or "negative_prompts" in data:
            neg_data: Any = (
                data["negative_prompt"]
                if "negative_prompt" in data
                else data["negative_prompts"]
            )
            companion.negative_prompt = (
                neg_data if isinstance(neg_data, list) else [neg_data]
            )

        # Parse positive_prompt
        if "positive_prompt" in data or "positive_prompts" in data:
            pos_data: Any = (
                data["positive_prompt"]
                if "positive_prompt" in data
                else data["positive_prompts"]
            )
            companion.positive_prompt = (
                pos_data if isinstance(pos_data, list) else [pos_data]
            )

        # Parse cfg
        if "clip_skip" in data:
            clip_skip_data: Any = data["clip_skip"]
            companion.clip_skip = (
                clip_skip_data if isinstance(clip_skip_data, list) else [clip_skip_data]
            )
            # Ensure all clip_skip values are negative
            companion.clip_skip = [-abs(int(cs)) for cs in companion.clip_skip]

        return companion
