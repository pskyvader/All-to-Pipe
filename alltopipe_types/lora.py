import os
from typing import Any
import folder_paths
import comfy.utils
import comfy.model_patcher
import comfy.sd
import torch
import logging

logger: logging.Logger = logging.getLogger("AllToPipe")


class LoraSpec:
    def __init__(
        self,
        name: str,
        subfolder: str,
        weight: float,
        clip_weight: float,
    ) -> None:
        self.name: str = name
        self.subfolder: str = subfolder
        self.weight: float = weight
        self.clip_weight: float = clip_weight
        self.cached_lora: Any | None = None


class LoraProcessor:
    @staticmethod
    def _infer_model_architecture(model_keys: set[str]) -> str | None:
        """Infer the model architecture from its keys.

        Returns the detected architecture name (SD15, SDXL, pony, illustrious, etc.)
        or None if architecture cannot be determined.
        """
        if not model_keys:
            return None

        key_count = len(model_keys)
        key_list = list(model_keys)[:100]  # Sample first 100 keys

        # Check for architecture-specific patterns in keys
        has_input_blocks = any("input_blocks" in key for key in key_list)
        has_diffusion_model = any("diffusion_model" in key for key in key_list)

        # SD1.5 has different key count than SDXL/XL models
        # SD1.5: ~600-700 keys, SDXL/XL: ~1600-2000+ keys
        if key_count < 1200:
            if has_input_blocks and has_diffusion_model:
                return "SD15"
        else:
            # XL-based models have more keys
            # We can't reliably distinguish them from keys alone
            return None

        return None

    @staticmethod
    def _normalize_architecture_name(name: str) -> str:
        """Normalize architecture names for comparison."""
        return name.lower() if name else ""

    @staticmethod
    def load_lora(lora: LoraSpec) -> dict[str, torch.Tensor]:
        if not lora.cached_lora:
            target_path = os.path.join(lora.subfolder, lora.name)
            lora_path = folder_paths.get_full_path("loras", target_path)

            if not lora_path:
                raise FileNotFoundError(f"LoRA '{target_path}' not found.")

            loaded_data = comfy.utils.load_torch_file(lora_path)
            lora.cached_lora = loaded_data
        return lora.cached_lora

    @staticmethod
    def get_model_key_set(model: comfy.model_patcher.ModelPatcher) -> set[str]:
        return set(model.model.state_dict().keys())

    @staticmethod
    def is_lora_compatible(
        lora_weights: dict[str, torch.Tensor], model_keys: set[str], lora: LoraSpec
    ) -> bool:
        """Check if LoRA is compatible with model based on architecture matching.

        A LoRA is compatible if:
        1. It has valid LoRA-formatted keys (lora_te_ or lora_unet_)
        2. The model and LoRA appear to be for compatible architectures
        """
        if not lora_weights or not model_keys:
            return False

        # 1. Check if LoRA has valid structure
        lora_keys_list = list(lora_weights.keys())
        has_te_keys = any(k.startswith("lora_te_") for k in lora_keys_list)
        has_unet_keys = any(k.startswith("lora_unet_") for k in lora_keys_list)

        if not (has_te_keys or has_unet_keys):
            return False

        # 2. Infer model architecture
        inferred_model_arch = LoraProcessor._infer_model_architecture(model_keys)
        lora_arch_norm = LoraProcessor._normalize_architecture_name(lora.subfolder)

        # 3. Compatibility logic based on architecture detection
        # If we can confidently detect SD1.5 model architecture
        if inferred_model_arch == "SD15":
            # Check if LoRA is also SD15-based
            # SD15 LoRAs have different structure than XL LoRAs
            # SD15 LoRAs typically have lora_te and lora_unet keys
            # XL LoRAs might have different patterns

            # For now, trust the subfolder comparison if visible
            if lora_arch_norm and lora_arch_norm != "sd15":
                return False

            # If we have both text encoder and unet keys, it's likely compatible
            # SD1.5 and SDXL can't be mixed, so Pony/Illustrious XL models won't work
            return True

        # If we can't determine model architecture (likely XL-based)
        # Do a heuristic check: XL models have many more keys
        model_key_count = len(model_keys)
        if model_key_count > 1200:
            # This is likely an XL-based model (SDXL, Pony, Illustrious, etc.)
            # XL-based LoRAs should have specific patterns we can't easily verify
            # So we'll be permissive and only reject if we're confident it's incompatible

            # If LoRA is labeled as SD15, it can't be used with XL models
            if lora_arch_norm == "sd15":
                return False

            # Otherwise assume compatible for XL-to-XL
            return True

        # Fallback: if we have valid LoRA keys, assume compatible
        return True

    @staticmethod
    def apply_lora(
        model: comfy.model_patcher.ModelPatcher,
        clip: comfy.sd.CLIP,
        loras: list[Any],
    ) -> tuple[comfy.model_patcher.ModelPatcher, comfy.sd.CLIP]:
        if not loras:
            return (model, clip)

        model_keys: set[str] = LoraProcessor.get_model_key_set(model)
        patched_model = model
        patched_clip = clip

        for lora in loras:
            try:
                lora_weights: dict[str, torch.Tensor] = LoraProcessor.load_lora(lora)

                # Perform compatibility check (preliminary - actual validation by ComfyUI API)
                if not LoraProcessor.is_lora_compatible(lora_weights, model_keys, lora):
                    message: str = (
                        f"LoRA '{lora.name}' may not be fully compatible with the model"
                    )
                    logger.warning(message)
                    # Don't skip - let ComfyUI API handle it

                patched_model, patched_clip = comfy.sd.load_bypass_lora_for_models(
                    patched_model,
                    patched_clip,
                    lora_weights,
                    lora.weight,
                    lora.clip_weight,
                )
            except Exception as e:
                logger.error(f"Failed to load {lora.name}: {e}")
                continue

        return (patched_model, patched_clip)
