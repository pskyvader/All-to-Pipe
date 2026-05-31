import torch
import math
from typing import Any


class PromptContainer:
    """Base container for managing prompt state and metadata."""

    def __init__(self) -> None:
        self.allow_missing: bool = True
        self.model: str | None = None
        self.lora: str | None = None


class PositivePrompt(PromptContainer):
    """Specific container for subject-focused positive features."""

    ALLOWED_FEATURES: tuple[str, ...] = (
        "characters",
        "age",
        "body",
        "race",
        "face",
        "hair",
        "clothes",
        "accessories",
        "location",
        "action",
        "pose",
        "camera",
        "lighting",
        "style",
        "color",
        "environment",
        "embeddings",
        "tags",
    )


class NegativePrompt(PromptContainer):
    """Specific container for exclusion-focused negative features."""

    ALLOWED_FEATURES: tuple[str, ...] = (
        "permanent",
        "embeddings",
        "style",
        "color",
        "tags",
    )


class PromptProcessor:
    """Handles multi-encoder synchronization, positional decay, and tensor aggregation."""

    DECAY_K: float = 0.0034055
    DECAY_FLOOR: float = 0.5

    @staticmethod
    def detect_architecture(clip: Any) -> str:
        """Identifies model type to determine metadata requirements."""
        test_tokens: dict[str, Any] = clip.tokenize("")
        keys: list[str] = list(test_tokens.keys())

        if "t5xxl" in keys:
            return "t5_hybrid"  # SD3, Flux
        if "g" in keys and "l" in keys:
            return "sdxl"  # SDXL, Pony
        return "sd15"  # SD1.5

    @staticmethod
    def get_tokenizer_limits(clip: Any) -> tuple[int, int]:
        """Retrieves max sequence length and internal chunking limits."""
        tokenizer: Any = (
            clip.tokenizer.clip_l
            if hasattr(clip.tokenizer, "clip_l")
            else clip.tokenizer
        )
        max_len: int = getattr(tokenizer, "max_length", 77)
        return int(max_len), int(max_len - 2)

    @staticmethod
    def get_special_tokens(
        token_data: dict[str, list[list[tuple[int, float]]]],
    ) -> tuple[int, int]:
        """Extracts start and end token IDs (SOH/EOH)."""
        first_key: str = next(iter(token_data))
        sample: list[tuple[int, float]] = token_data[first_key][0]
        return int(sample[0][0]), int(sample[-1][0])

    @staticmethod
    def clean_token_streams(
        token_data: dict[str, list[list[tuple[int, float]]]], start_id: int, end_id: int
    ) -> dict[str, list[tuple[int, float]]]:
        """Removes delimiters and padding to prepare raw token streams."""
        return {
            k: [t for t in v[0] if t[0] not in (start_id, end_id, 0)]
            for k, v in token_data.items()
        }

    @classmethod
    def apply_decay_to_segment(
        cls, segment: list[tuple[int, float]], start_offset: int
    ) -> list[tuple[int, float]]:
        """Calculates and applies the exponential weight decay to a chunk segment."""
        decayed: list[tuple[int, float]] = []
        for i, (token_id, weight) in enumerate(segment):
            global_pos: int = start_offset + i
            multiplier: float = cls.DECAY_FLOOR + (1.0 - cls.DECAY_FLOOR) * math.exp(
                -cls.DECAY_K * global_pos
            )
            decayed.append((token_id, weight * multiplier))
        return decayed

    @staticmethod
    def wrap_and_pad_block(
        streams: dict[str, list[tuple[int, float]]],
        start_id: int,
        end_id: int,
        max_len: int,
    ) -> dict[str, list[list[tuple[int, float]]]]:
        """Re-inserts delimiters and pads streams to the required CLIP sequence length."""
        block: dict[str, list[list[tuple[int, float]]]] = {}
        for k, tokens in streams.items():
            formatted: list[tuple[int, float]] = (
                [(start_id, 1.0)] + tokens + [(end_id, 1.0)]
            )
            if len(formatted) < max_len:
                formatted += [(0, 0.0)] * (max_len - len(formatted))
            block[k] = [formatted[:max_len]]
        return block

    @staticmethod
    def extract_pooled_output(encoded_result: list[Any]) -> torch.Tensor:
        """Safely extracts the pooled_output tensor from the CLIP encoding result."""
        # result format: [[tensor, {"pooled_output": tensor}]]
        data: dict[str, Any] | torch.Tensor = encoded_result[0][1]
        if isinstance(data, dict) and "pooled_output" in data:
            return data["pooled_output"]
        return data  # Fallback for SD1.5/Simple encoders

    @classmethod
    def encode_prompt(
        cls,
        clip: Any,
        text: str,
        width: int,
        height: int,
        target_width: int,
        target_height: int,
        crop_w: int,
        crop_h: int,
    ) -> list[list[torch.Tensor | dict[str, Any]]]:
        """
        Main entry point. Synchronizes multi-encoder tokens and aggregates tensors.
        Explicitly raises ValueError on empty strings to prevent downstream sampler errors.
        """
        if not text.strip():
            raise ValueError("Prompt text cannot be empty.")

        # 1. Initialization and Limit Detection
        model_type: str = cls.detect_architecture(clip)
        max_len, chunk_limit = cls.get_tokenizer_limits(clip)

        # 2. Tokenization and Cleaning
        token_data: dict[str, list[list[tuple[int, float]]]] = clip.tokenize(text)
        start_id, end_id = cls.get_special_tokens(token_data)
        clean_streams: dict[str, list[tuple[int, float]]] = cls.clean_token_streams(
            token_data, start_id, end_id
        )

        # 3. Synchronized Processing
        ref_key: str = "l" if "l" in clean_streams else next(iter(clean_streams))
        num_tokens: int = len(clean_streams[ref_key])
        num_chunks: int = max(1, math.ceil(num_tokens / chunk_limit))

        cond_list: list[torch.Tensor] = []
        pooled_list: list[torch.Tensor] = []

        for i in range(num_chunks):
            start_idx: int = i * chunk_limit
            end_idx: int = (i + 1) * chunk_limit
            chunk_content: dict[str, list[tuple[int, float]]] = {}

            for k in clean_streams:
                # Ensure all encoders (G, L, T5) are sliced at the exact same text index
                segment: list[tuple[int, float]] = clean_streams[k][start_idx:end_idx]
                chunk_content[k] = cls.apply_decay_to_segment(segment, start_idx)

            # Format and Encode
            formatted_block: dict[str, Any] = cls.wrap_and_pad_block(
                chunk_content, start_id, end_id, max_len
            )
            encoded: list[Any] = clip.encode_from_tokens_scheduled(formatted_block)

            cond_list.append(encoded[0][0])
            pooled_list.append(cls.extract_pooled_output(encoded))

        # 4. Final Aggregation and Metadata Injection
        full_cond: torch.Tensor = torch.cat(cond_list, dim=1)

        # Pooled output from the first chunk represents the primary context
        metadata: dict[str, Any] = {"pooled_output": pooled_list[0]}

        # Micro-conditioning for SDXL architecture
        if model_type == "sdxl":
            metadata.update(
                {
                    "width": width,
                    "height": height,
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                    "target_width": target_width,
                    "target_height": target_height,
                }
            )

        return [[full_cond, metadata]]
