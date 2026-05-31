"""
All-to-Pipe parameters builder node.

Builds and attaches sampler parameters to the Pipe.
"""

from typing import Any
import random
from ..alltopipe_types import Pipe, Parameters

# from ..common.utils import deep_copy_pipe
from ..common.constants import (
    SUPPORTED_SAMPLERS,
    SUPPORTED_SCHEDULERS,
    MIN_STEPS,
    MAX_STEPS,
    MIN_CFG,
    MAX_CFG,
    MIN_SEED,
    MAX_SEED,
)


class ParametersBuilderNode:
    """
    Builds and attaches sampler parameters to the Pipe.

    Features:
    - COMBO selectors for sampler and scheduler
    - Random seed generation option
    - Validation of all parameter ranges
    """

    def __init__(self) -> None:
        """Initialize the parameters builder node."""
        pass

    @staticmethod
    def execute(
        pipe: Pipe | None = None,
        steps: int = 20,
        cfg: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = 0,
        denoise: float = 1.0,
    ) -> tuple[Pipe]:
        """
        Execute the node and build parameters for the pipe.

        Args:
            pipe: Optional Pipe instance (creates new if None)
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler: Sampler name or 'RANDOM' for random selection
            scheduler: Scheduler name or 'RANDOM' for random selection
            seed: Random seed for reproducibility

        Returns:
            Tuple containing the modified Pipe instance
        """
        # Deep copy pipe to avoid modifying the original
        # new_pipe: Pipe = deep_copy_pipe(pipe) if pipe is not None else Pipe()
        new_pipe: Pipe = pipe.clone() if pipe is not None else Pipe()

        # Handle RANDOM sampler selection
        selected_sampler = sampler
        if sampler == "RANDOM":
            selected_sampler = random.choice(SUPPORTED_SAMPLERS)

        # Handle RANDOM scheduler selection
        selected_scheduler = scheduler
        if scheduler == "RANDOM":
            selected_scheduler = random.choice(SUPPORTED_SCHEDULERS)

        # Create and validate the parameters
        parameters: Parameters = Parameters(
            steps=steps,
            cfg=cfg,
            sampler=selected_sampler,
            scheduler=selected_scheduler,
            seed=seed,
            denoise=denoise,
        )
        # validate_parameters(parameters)

        # Attach parameters to the pipe
        new_pipe.parameters = parameters

        return (new_pipe,)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """
        Define the input types for this node with dynamic selectors.

        Returns:
            Dictionary defining node inputs with COMBO selectors
        """
        # Add RANDOM option at the beginning of sampler and scheduler lists
        sampler_options = ["RANDOM"] + SUPPORTED_SAMPLERS
        scheduler_options = ["RANDOM"] + SUPPORTED_SCHEDULERS

        return {
            "optional": {
                "pipe": ("PIPE",),
            },
            "required": {
                "steps": ("INT", {"default": 20, "min": MIN_STEPS, "max": MAX_STEPS}),
                "cfg": (
                    "FLOAT",
                    {"default": 7.0, "min": MIN_CFG, "max": MAX_CFG, "step": 0.1},
                ),
                "sampler": (tuple(sampler_options),),
                "scheduler": (tuple(scheduler_options),),
                "seed": ("INT", {"default": 0, "min": MIN_SEED, "max": MAX_SEED}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 1, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES: tuple[str, ...] = ("PIPE",)
    RETURN_NAMES: tuple[str, ...] = ("pipe",)
    FUNCTION: str = "execute"
    CATEGORY: str = "all-to-pipe"
