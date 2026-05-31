from typing import Any
import random
import torch
import comfy.model_management


class ImageConfig:
    def __init__(
        self,
        width: int,
        height: int,
        batch_size: int,
        noise: float = 1.0,
        color_code: str | None = None,
        image: torch.Tensor | None = None,
        latent: dict[str, torch.Tensor | None] | None = None,
    ) -> None:
        """
        Image configuration for generation.
        Specifies dimensions, batch size, and noise parameters.
        """
        self.width: int = width
        self.height: int = height
        self.batch_size: int = batch_size
        self.noise: float = max(0.0, min(1.0, noise))
        self.color_code: str | None = color_code
        self.image: torch.Tensor | None = image
        self.latent: dict[str, torch.Tensor | None] | None = latent


class ImageConfigProcessor:
    @staticmethod
    def create_noisy_image(image_config: ImageConfig, seed: int | None) -> torch.Tensor:
        """
        Create a batch of noisy images based on config (Pixel Space).
        """
        if image_config.width <= 0 or image_config.height <= 0:
            raise ValueError("Width and height must be positive")

        if image_config.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if seed:
            torch.manual_seed(seed)
            random.seed(seed)
        device: torch.device = comfy.model_management.get_torch_device()
        if not isinstance(device, torch.device):
            raise ValueError("Device Not valid")

        color_vec: torch.Tensor = ImageConfigProcessor.get_color_from_code(
            image_config.color_code, device="cpu"
        )

        color_bias: torch.Tensor = (
            color_vec.view(1, 3, 1, 1)
            .expand(image_config.batch_size, 3, image_config.height, image_config.width)
            .to(device)
        )

        noise_tensor: torch.Tensor = torch.randn(
            (image_config.batch_size, 3, image_config.height, image_config.width),
            device=device,
        )

        noise_tensor = torch.clamp((noise_tensor * 0.5) + 0.5, 0.0, 1.0)

        image_samples: torch.Tensor = (color_bias * (1.0 - image_config.noise)) + (
            noise_tensor * image_config.noise
        )

        return image_samples.permute(0, 2, 3, 1)

    def get_color_from_code(
        color_code: str | None, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Parses hex color and returns a normalized 3-channel RGB vector.
        """
        if color_code is None:
            r, g, b = [random.randint(0, 255) for _ in range(3)]
        else:
            hex_val: str = color_code.strip().lstrip("#")

            if len(hex_val) == 3:
                hex_val = "".join([c * 2 for c in hex_val])

            if len(hex_val) != 6:
                raise ValueError(f"Invalid hex color: {color_code}")

            try:
                r: int = int(hex_val[0:2], 16)
                g: int = int(hex_val[2:4], 16)
                b: int = int(hex_val[4:6], 16)
            except ValueError:
                raise ValueError(f"Could not parse hex color: {color_code}")

        return torch.tensor(
            [r / 255.0, g / 255.0, b / 255.0], dtype=torch.float32, device=device
        )
