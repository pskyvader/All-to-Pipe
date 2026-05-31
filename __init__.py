"""
All-to-Pipe module for ComfyUI.

A custom node module to build, manipulate, and export reusable generation pipelines
containing models, LoRAs, parameters, and prompt structures.
"""

from typing import Any

from .nodes.update_pipe_node import UpdatePipeNode
from .nodes.positive_prompt_node import PositivePromptNode
from .nodes.negative_prompt_node import NegativePromptNode
from .nodes.model_node import ModelNode
from .nodes.lora_node import LoraNode
from .nodes.parameters_builder_node import ParametersBuilderNode
from .nodes.image_config_node import ImageConfigNode
from .nodes.template_node import TemplateNode
from .nodes.export_node import ExportNode
from .nodes.export_json_node import ExportJsonNode
from .nodes.export_text_node import ExportTextNode
from .nodes.export_single_node import ExportSingleNode

# Node class mappings for ComfyUI registration
NODE_CLASS_MAPPINGS: dict[str, type[Any]] = {
    "AllToPipe_Update": UpdatePipeNode,
    "AllToPipe_ExportSingle": ExportSingleNode,
    "AllToPipe_PositivePrompt": PositivePromptNode,
    "AllToPipe_NegativePrompt": NegativePromptNode,
    "AllToPipe_Model": ModelNode,
    "AllToPipe_LoRA": LoraNode,
    "AllToPipe_Parameters": ParametersBuilderNode,
    "AllToPipe_ImageConfig": ImageConfigNode,
    "AllToPipe_Template": TemplateNode,
    "AllToPipe_SamplerExport": ExportNode,
    "AllToPipe_JsonExport": ExportJsonNode,
    "AllToPipe_TextExport": ExportTextNode,
}

# Human-readable display name mappings
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "AllToPipe_Update": "Update Pipe",
    "AllToPipe_ExportSingle": "Export single",
    "AllToPipe_PositivePrompt": "Positive Prompt",
    "AllToPipe_NegativePrompt": "Negative Prompt",
    "AllToPipe_Model": "Model",
    "AllToPipe_LoRA": "LoRA",
    "AllToPipe_Parameters": "Parameters",
    "AllToPipe_ImageConfig": "Image Config",
    "AllToPipe_Template": "Template",
    "AllToPipe_SamplerExport": "Export (Sampler)",
    "AllToPipe_JsonExport": "Export (JSON)",
    "AllToPipe_TextExport": "Export (Text)",
}

# CRITICAL: Register custom types used by All-to-Pipe nodes
# This is required for ComfyUI to understand and validate node connections
CUSTOM_TYPE_NAMES = [
    "PIPE",
]
