import json
from typing import Any
from pathlib import Path
from ..common.prompt_helpers import prompt_to_string

from ..alltopipe_types import (
    Pipe,
    Model,
    Parameters,
    ImageConfig,
    PositivePrompt,
    NegativePrompt,
    Template,
    TemplateParser,
)


class ExportTextNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "pipe": ("PIPE",),
            },
            "optional": {
                "model": ("BOOLEAN", {"default": True}),
                "loras": ("BOOLEAN", {"default": True}),
                "parameters": ("BOOLEAN", {"default": True}),
                "image_config": ("BOOLEAN", {"default": True}),
                "prompt_text": ("BOOLEAN", {"default": True}),
                "prompt_map": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "FLOAT",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "INT",
        "FLOAT",
        "STRING",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "INT",
        "FLOAT",
    )

    RETURN_NAMES = (
        "model_name",
        "model_subfolder",
        "lora_list",
        "first_lora_name",
        "first_lora_weight",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "seed",
        "denoise",
        "positive_prompt",
        "negative_prompt",
        "positive_map_json",
        "negative_map_json",
        "width",
        "height",
        "batch_size",
        "image_noise",
    )

    FUNCTION = "execute"
    CATEGORY = "all-to-pipe"

    @staticmethod
    def execute(
        pipe: Pipe,
        model: bool = True,
        loras: bool = True,
        parameters: bool = True,
        image_config: bool = True,
        prompt_text: bool = True,
        prompt_map: bool = True,
    ) -> tuple[
        str,
        str,
        str,
        str,
        float,
        int,
        float,
        str,
        str,
        int,
        float,
        str,
        str,
        str,
        str,
        int,
        int,
        int,
        float,
    ]:

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
        # 1. Model Data
        output_model_name = (
            (Path(pipe.model.name).stem if pipe.model else "") if model else ""
        )
        output_model_subfolder = (
            (pipe.model.subfolder if pipe.model else "") if model else ""
        )

        # 2. LoRA Data
        output_lora_list = ""
        first_lora_name = ""
        first_lora_weight = 0.0
        if loras and pipe.loras:
            # Serializing the lora objects into a JSON string list for easy parsing elsewhere
            lora_data: list[dict[str, Any]] = [
                {
                    "name": Path(lora.name).stem,
                    "subfolder": lora.subfolder,
                    "weight": lora.weight,
                    "clip_weight": lora.clip_weight,
                }
                for lora in pipe.loras
            ]
            output_lora_list = json.dumps(lora_data)
            first_lora_name = Path(lora_data[0]["name"]).stem
            first_lora_weight = lora_data[0]["weight"]

        # 3. Parameters
        output_steps = (
            (pipe.parameters.steps if pipe.parameters else -1) if parameters else -1
        )
        output_cfg = (
            (pipe.parameters.cfg if pipe.parameters else -1.0) if parameters else -1.0
        )
        output_sampler = (
            (pipe.parameters.sampler if pipe.parameters else "") if parameters else ""
        )
        output_scheduler = (
            (pipe.parameters.scheduler if pipe.parameters else "") if parameters else ""
        )
        output_seed = (
            (pipe.parameters.seed if pipe.parameters else -1) if parameters else -1
        )
        output_denoise = (
            (pipe.parameters.denoise if pipe.parameters else -1) if parameters else -1
        )

        # 4. Prompt Text (Rendered Strings)
        output_positive_prompt = ""
        output_negative_prompt = ""
        if prompt_text:
            if pipe.positive_template.parsed_template is None:
                pipe.positive_template.parsed_template = TemplateParser.parse_template(
                    pipe.positive_template.text,
                    pipe.positive_prompt,
                    pipe.positive_template.allow_missing,
                )
            output_positive_prompt = pipe.positive_template.parsed_template

            if pipe.negative_template.parsed_template is None:
                pipe.negative_template.parsed_template = TemplateParser.parse_template(
                    pipe.negative_template.text,
                    pipe.negative_prompt,
                    pipe.negative_template.allow_missing,
                )
            output_negative_prompt = pipe.negative_template.parsed_template

        # 5. Prompt Map (Converted to JSON Strings)
        output_positive_map_json = "{}"
        output_negative_map_json = "{}"
        if prompt_map:
            pos_data = prompt_to_string(pipe.positive_prompt)
            neg_data = prompt_to_string(pipe.negative_prompt)

            # Ensure we return a valid JSON string even if empty
            output_positive_map_json = json.dumps(pos_data) if pos_data else "{}"
            output_negative_map_json = json.dumps(neg_data) if neg_data else "{}"

        # 6. Image Config
        output_width = (
            (pipe.image_config.width if pipe.image_config else -1)
            if image_config
            else -1
        )
        output_height = (
            (pipe.image_config.height if pipe.image_config else -1)
            if image_config
            else -1
        )
        output_batch_size = (
            (pipe.image_config.batch_size if pipe.image_config else -1)
            if image_config
            else -1
        )
        output_image_noise = (
            (pipe.image_config.noise if pipe.image_config else -1)
            if image_config
            else -1
        )

        return (
            output_model_name,
            output_model_subfolder,
            output_lora_list,
            first_lora_name,
            first_lora_weight,
            output_steps,
            output_cfg,
            output_sampler,
            output_scheduler,
            output_seed,
            output_denoise,
            output_positive_prompt,
            output_negative_prompt,
            output_positive_map_json,
            output_negative_map_json,
            output_width,
            output_height,
            output_batch_size,
            output_image_noise,
        )
