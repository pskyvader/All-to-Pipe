"""
All-to-Pipe constants module.

Centralized constant definitions used by validators and nodes.
"""

from typing import Final

# Import samplers and schedulers directly from ComfyUI source
from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES

SUPPORTED_SAMPLERS: Final[list[str]] = list(SAMPLER_NAMES)
SUPPORTED_SCHEDULERS: Final[list[str]] = list(SCHEDULER_NAMES)

# Model subfolder paths
DEFAULT_MODEL_SUBFOLDER: Final[str] = ""
DEFAULT_LORA_SUBFOLDER: Final[str] = ""

# Validation ranges
MIN_STEPS: Final[int] = 1
MAX_STEPS: Final[int] = 100

MIN_CFG: Final[float] = 0.0
MAX_CFG: Final[float] = 20.0

MIN_SEED: Final[int] = 0
MAX_SEED: Final[int] = 0xFFFFFFFF

MIN_LORA_WEIGHT: Final[float] = -5.0
MAX_LORA_WEIGHT: Final[float] = 5.0

MIN_CLIP_WEIGHT: Final[float] = -5.0
MAX_CLIP_WEIGHT: Final[float] = 5.0
