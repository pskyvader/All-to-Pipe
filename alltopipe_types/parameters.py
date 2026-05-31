"""
Parameters type and processor for All-to-Pipe.

Handles generation parameters and sampler configuration.
"""

from typing import Any


class Parameters:
    """
    Explicit sampler-related parameters.
    No defaults are allowed - all fields must be explicitly set.
    """

    def __init__(
        self,
        steps: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        seed: int,
        denoise: float,
    ) -> None:
        """
        Initialize generation parameters.

        Args:
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler: Sampler name (e.g., 'euler', 'dpmpp_2m')
            scheduler: Scheduler name (e.g., 'normal', 'karras')
            seed: Random seed for reproducibility
        """
        self.steps: int = steps
        self.cfg: float = cfg
        self.sampler: str = sampler
        self.scheduler: str = scheduler
        self.seed: int = seed
        self.denoise: float = denoise


class ParametersProcessor:
    """
    Processor for Parameters operations.
    Handles export and formatting of sampler parameters.
    """

    @staticmethod
    def export_sampler_params(parameters: Parameters) -> dict[str, Any]:
        """
        Export parameters as a dictionary ready for KSampler.

        Args:
            parameters: Parameters instance to export

        Returns:
            Dictionary with KSampler-compatible parameter keys

        Raises:
            ValueError: If parameters is None or invalid
        """

        return {
            "steps": parameters.steps,
            "cfg": parameters.cfg,
            "sampler_name": parameters.sampler,
            "scheduler": parameters.scheduler,
            "seed": parameters.seed,
            "denoise": parameters.denoise,
        }
