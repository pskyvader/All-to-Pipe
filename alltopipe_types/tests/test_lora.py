"""
Unit tests for LoRA loading, compatibility checking, and application.

Validates that LoraSpec correctly handles all variables and that
architecture matching works across a sampled matrix of real files.
"""

import pytest
import sys
import itertools
import logging
import gc
import torch
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alltopipe_types.lora import LoraSpec, LoraProcessor
from alltopipe_types.model import Model, ModelProcessor

logger: logging.Logger = logging.getLogger("AllToPipe_Test")
# Define base paths for ComfyUI models
CHECKPOINT_BASE = Path(r"E:\ComfyUI\models\checkpoints")
LORA_BASE = Path(r"E:\ComfyUI\models\loras")


def get_all_test_combinations():
    """
    Discovers a SAMPLE of models and loras to prevent out-of-memory crashes.
    Returns: (combinations_list, ids_list)
    """
    all_models = []
    if CHECKPOINT_BASE.exists():
        for arch_dir in CHECKPOINT_BASE.iterdir():
            if arch_dir.is_dir():
                sample_files = list(arch_dir.rglob("*.safetensors"))[:2]
                for file in sample_files:
                    all_models.append(
                        Model(name=file.name, subfolder=arch_dir.name, clip_skip=-1)
                    )

    all_loras = []
    if LORA_BASE.exists():
        for arch_dir in LORA_BASE.iterdir():
            if arch_dir.is_dir():
                sample_files = list(arch_dir.rglob("*.safetensors"))[:2]
                for file in sample_files:
                    all_loras.append(
                        LoraSpec(
                            name=file.name,
                            subfolder=arch_dir.name,
                            weight=1.0,
                            clip_weight=0.8,
                        )
                    )

    combinations = []
    ids = []
    for m, l in itertools.product(all_models, all_loras):
        expected = m.subfolder.lower() == l.subfolder.lower()
        combinations.append((m, l, expected))
        # This creates the readable name in your test output
        ids.append(f"{m.subfolder}/{m.name}_vs_{l.subfolder}/{l.name}")

    return combinations, ids


# Unpack the data and the IDs for use in the decorator
test_combinations, test_ids = get_all_test_combinations()


class TestLoraSpec:
    """Test LoraSpec initialization and attribute verification."""

    def test_lora_spec_creation(self) -> None:
        """Verify that all variables are correctly assigned to the LoraSpec instance."""
        name = "test_lora.safetensors"
        subfolder = "SDXL"
        weight = 0.75
        clip_weight = 0.42

        lora = LoraSpec(
            name=name, subfolder=subfolder, weight=weight, clip_weight=clip_weight
        )

        assert lora.name == name
        assert lora.subfolder == subfolder
        assert lora.weight == weight
        assert lora.clip_weight == clip_weight
        assert lora.cached_lora is None

    def test_lora_spec_caching(self) -> None:
        lora = LoraSpec(
            name="test.safetensors", subfolder="SD15", weight=1.0, clip_weight=1.0
        )
        mock_data = {"layer1": "tensor_data"}
        lora.cached_lora = mock_data
        assert lora.cached_lora == mock_data


class TestLoraProcessor:
    """Test LoRA logic using a sampled matrix of real files."""

    @pytest.mark.parametrize(
        "model_spec, lora_spec, expected", test_combinations, ids=test_ids
    )
    def test_lora_compatibility_matrix(
        self, model_spec: Model, lora_spec: LoraSpec, expected: bool
    ) -> None:
        """Validates architecture matching while ensuring specs are fully populated."""
        loaded = None
        model = None

        try:
            # Verify weight values are actually present in the spec before testing
            assert isinstance(lora_spec.weight, float)
            assert isinstance(lora_spec.clip_weight, float)

            # 1. Load Model
            loaded = ModelProcessor.load_model(model_spec)
            assert loaded is not None, f"Failed to load: {model_spec.name}"
            model, clip, vae = loaded

            # 2. Extract Data
            model_keys = LoraProcessor.get_model_key_set(model)
            lora_weights = LoraProcessor.load_lora(lora_spec)

            # 3. Check Compatibility
            result = LoraProcessor.is_lora_compatible(
                lora_weights, model_keys, lora_spec
            )

            # 4. Assert
            assert result is expected, (
                f"\nCOMPATIBILITY LOGIC FAILED!\n"
                f"Model: {model_spec.subfolder} ({model_spec.name})\n"
                f"LoRA:  {lora_spec.subfolder} ({lora_spec.name})\n"
                f"Weights: strength={lora_spec.weight}, clip={lora_spec.clip_weight}\n"
                f"Returned {result}, but we expected {expected}."
            )

            logger.info(
                f"Verified: {model_spec.subfolder} x {lora_spec.subfolder} -> {result}"
            )

        finally:
            del model, loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def test_load_lora_not_found(self) -> None:
        lora = LoraSpec(
            name="missing.safetensors", subfolder="SD15", weight=1.0, clip_weight=1.0
        )
        with pytest.raises(FileNotFoundError):
            LoraProcessor.load_lora(lora)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-o", "log_cli=true", "--log-cli-level=INFO"])
