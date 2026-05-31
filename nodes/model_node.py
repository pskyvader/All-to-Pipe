"""
All-to-Pipe model node.

Assigns a model to an existing Pipe with companion file loading and parameter adjustment.
"""

from typing import Any
import os
import random

from custom_nodes.all_to_pipe.common.companion_loader import CompanionFile
from ..alltopipe_types import (
    Pipe,
    Model,
    PositivePrompt,
    NegativePrompt,
    ModelProcessor,
)

# from ..common.utils import deep_copy_pipe
from ..common.file_helpers import (
    discover_models_in_subfolder,
)
from ..common.companion_loader import CompanionLoader


class ModelNode:
    """
    Assigns a model to an existing Pipe with dynamic subfolder and model selection.

    Features:
    - COMBO selector for model subfolders
    - COMBO selector for available models in subfolder
    - Random model selection
    - Loads and applies companion JSON file limits
    - Parses and distributes prompt data
    """

    def __init__(self) -> None:
        """Initialize the model node."""
        pass

    @staticmethod
    def execute(
        pipe: Pipe | None = None,
        model_selection: str = "",
        load_companion: bool = False,
        random_subfolder: str = "all",
        clip_skip: int = -1,
    ) -> tuple[Pipe]:
        """
        Execute the node and assign a model to the pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            model_selection: Model selection (either "RANDOM /" or "subfolder/model_name.ext")
            load_companion: Whether to load and apply companion file data
            random_subfolder: Subfolder to randomly select from when "RANDOM /" is chosen

        Returns:
            Tuple containing the modified Pipe instance
        """
        # new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()
        new_pipe: Pipe = pipe.clone() if pipe is not None else Pipe()

        if not new_pipe.parameters:
            raise ValueError("Pipe Needs Parameters before loading a model")
        if not new_pipe.image_config:
            raise ValueError("Pipe Needs Image Config before loading a model")

        # Handle RANDOM selection
        if model_selection == "RANDOM /":
            # Get models from specified subfolder or all subfolders
            if random_subfolder == "all":
                all_models = ModelNode._get_all_models()
            else:
                all_models = discover_models_in_subfolder(random_subfolder)

            if not all_models:
                raise ValueError(
                    f"No models found in subfolder: {random_subfolder if random_subfolder != 'all' else 'any'}"
                )
            model_selection = (
                random.choice(all_models)
                if random_subfolder == "all"
                else f"{random_subfolder}/{random.choice(all_models)}"
            )

        # Parse model selection string (format: "subfolder/model_name.ext" or "model_name.ext")
        if "/" in model_selection:
            parts = model_selection.rsplit(
                "/", 1
            )  # Split from right to handle subfolders with /
            model_subfolder = parts[0]
            model_name = parts[1]
        else:
            model_subfolder = ""
            model_name = model_selection

        # Get available models in subfolder for validation
        available_models = discover_models_in_subfolder(model_subfolder)

        if not available_models:
            raise ValueError(f"No models found in subfolder: {model_subfolder}")

        if model_name not in available_models:
            raise ValueError(
                f"Model {model_name} not found in {model_subfolder or 'root folder'}"
            )

        # Create and attach the model
        new_pipe.model = Model(
            name=model_name, subfolder=model_subfolder, clip_skip=clip_skip
        )
        new_pipe.model.cached_model = ModelProcessor.load_model(new_pipe.model)

        companion: CompanionFile | None = (
            CompanionLoader.load_model_companion(model_name, model_subfolder)
            if load_companion
            else None
        )

        if companion is not None:
            # Store companion file as dictionary for later reference
            new_pipe.companion_model_data = companion.raw_data

            if companion.positive_prompt:
                if new_pipe.positive_prompt is None:
                    new_pipe.positive_prompt = PositivePrompt()
                new_pipe.positive_prompt.model = CompanionLoader.apply_text_suggestions(
                    companion.positive_prompt,
                    "",
                    "Positive Prompts",
                )
                if new_pipe.positive_template:
                    new_pipe.positive_template.parsed_template = None

            if companion.negative_prompt:
                if new_pipe.negative_prompt is None:
                    new_pipe.negative_prompt = NegativePrompt()
                new_pipe.negative_prompt.model = CompanionLoader.apply_text_suggestions(
                    companion.negative_prompt,
                    "",
                    "Negative Prompts",
                )
                if new_pipe.negative_template:
                    new_pipe.negative_template.parsed_template = None

            if new_pipe.parameters:
                new_pipe.parameters = CompanionLoader.apply_companion_to_parameters(
                    companion, new_pipe.parameters
                )
            if companion.resolution:
                image_config = CompanionLoader.apply_companion_to_image_config(
                    companion, new_pipe.image_config
                )

                # image_config.image = ImageConfigProcessor.create_noisy_image(
                #     image_config,
                #     new_pipe.parameters.seed if new_pipe.parameters else None,
                # )

                if image_config.image is not None:
                    _, _, vae = new_pipe.model.cached_model
                    image_config.latent = {
                        "samples": vae.encode(image_config.image[:, :, :, :3])
                    }
                new_pipe.image_config = image_config

            if companion.clip_skip:
                print(
                    f"companion model applying clip skip:{companion.clip_skip} , to model:{new_pipe.model.name}"
                )
                new_pipe.model = CompanionLoader.apply_companion_to_model(
                    companion, new_pipe.model
                )

        return (new_pipe,)

    @staticmethod
    def _get_all_models(base_path: str = "models/checkpoints") -> list[str]:
        """
        Recursively discover all models in all subfolders.

        Args:
            base_path: Base path to checkpoints directory

        Returns:
            List of model paths in format "subfolder/model_name.ext" or "model_name.ext"
        """
        models: list[str] = []
        model_extensions: tuple[str, ...] = (".safetensors", ".ckpt", ".pt", ".pth")

        if not os.path.isdir(base_path):
            return models

        try:
            # Walk all subdirectories
            for root, _, files in os.walk(base_path):
                for filename in files:
                    if any(filename.lower().endswith(ext) for ext in model_extensions):
                        # Get relative path from base
                        rel_path = os.path.relpath(
                            os.path.join(root, filename), base_path
                        )
                        # Normalize path separators to forward slash
                        rel_path = rel_path.replace(os.sep, "/")
                        models.append(rel_path)
        except (OSError, PermissionError):
            pass

        models.sort()
        return models

    @staticmethod
    def _get_model_subfolders(base_path: str = "models/checkpoints") -> list[str]:
        """
        Get all available model subfolders.

        Args:
            base_path: Base path to checkpoints directory

        Returns:
            List of subfolder names
        """
        subfolders: list[str] = ["all"]  # Include "all" option

        if not os.path.isdir(base_path):
            return subfolders

        try:
            for item in os.listdir(base_path):
                path = os.path.join(base_path, item)
                if os.path.isdir(path):
                    subfolders.append(item)
        except (OSError, PermissionError):
            pass

        # subfolders.sort()
        return subfolders

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """
        Define the input types for this node with improved selectors.

        Returns:
            Dictionary defining node inputs with all models discovered recursively
        """
        # Get all models recursively from all subfolders
        all_models = cls._get_all_models()

        # Get available subfolders for random selection
        model_subfolders = cls._get_model_subfolders()

        # Add RANDOM option
        model_options = ["RANDOM /"] + all_models if all_models else ["RANDOM /"]
        default_model = all_models[0] if all_models else "RANDOM /"

        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "model_selection": (
                    (model_options,)
                    if model_options
                    else ("STRING", {"default": default_model})
                ),
                "load_companion": ("BOOLEAN", {"default": True}),
                "random_subfolder": (
                    (model_subfolders,)
                    if model_subfolders
                    else ("STRING", {"default": "all"})
                ),
                "clip_skip": (
                    "INT",
                    {"default": -1, "min": -24, "max": -1, "step": 1, "advanced": True},
                ),
            },
        }

    RETURN_TYPES: tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
