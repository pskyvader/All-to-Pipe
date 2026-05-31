"""
All-to-Pipe file helpers module.

Helper functions for resolving model and LoRA file paths and discovering available resources.
"""

import os

# Cache for discovered subfolders
_model_subfolder_cache: list[str] | None = None
_lora_subfolder_cache: list[str] | None = None


def discover_model_subfolders(
    base_path: str = "models/checkpoints",
    include_root: bool = True,
) -> list[str]:
    """
    Discover all subfolders in the models/checkpoints directory.

    Results are cached to avoid repeated filesystem scans.

    Args:
        base_path: Base path to checkpoints directory
        include_root: Whether to include empty string for root folder

    Returns:
        List of subfolder paths, sorted alphabetically.
        Empty string "" represents the root folder.
    """
    global _model_subfolder_cache

    if _model_subfolder_cache is not None:
        return _model_subfolder_cache

    subfolders: list[str] = []

    if include_root:
        subfolders.append("")

    # Check if base path exists
    if not os.path.isdir(base_path):
        _model_subfolder_cache = subfolders
        return subfolders

    # Scan for subfolders
    try:
        for item in os.listdir(base_path):
            item_path: str = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
    except (OSError, PermissionError):
        pass

    subfolders.sort()
    _model_subfolder_cache = subfolders
    return subfolders


def discover_lora_subfolders(
    base_path: str = "models/loras",
    include_root: bool = True,
) -> list[str]:
    """
    Discover all subfolders in the models/loras directory.

    Results are cached to avoid repeated filesystem scans.

    Args:
        base_path: Base path to loras directory
        include_root: Whether to include empty string for root folder

    Returns:
        List of subfolder paths, sorted alphabetically.
        Empty string "" represents the root folder.
    """
    global _lora_subfolder_cache

    if _lora_subfolder_cache is not None:
        return _lora_subfolder_cache

    subfolders: list[str] = []

    if include_root:
        subfolders.append("")

    # Check if base path exists
    if not os.path.isdir(base_path):
        _lora_subfolder_cache = subfolders
        return subfolders

    # Scan for subfolders
    try:
        for item in os.listdir(base_path):
            item_path: str = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
    except (OSError, PermissionError):
        pass

    subfolders.sort()
    _lora_subfolder_cache = subfolders
    return subfolders


def discover_models_in_subfolder(
    subfolder: str = "",
    base_path: str = "models/checkpoints",
) -> list[str]:
    """
    Discover all model files in a specific subfolder.

    Args:
        subfolder: Subfolder name (empty string for root)
        base_path: Base path to checkpoints directory

    Returns:
        List of model filenames, sorted alphabetically
    """
    models: list[str] = []

    full_path: str = os.path.join(base_path, subfolder) if subfolder else base_path

    if not os.path.isdir(full_path):
        return models

    # Common model extensions
    model_extensions: tuple[str, ...] = (".safetensors", ".ckpt", ".pt", ".pth")

    try:
        for item in os.listdir(full_path):
            if any(item.lower().endswith(ext) for ext in model_extensions):
                models.append(item)
    except (OSError, PermissionError):
        pass

    models.sort()
    return models


def discover_loras_in_subfolder(
    subfolder: str = "",
    base_path: str = "models/loras",
) -> list[str]:
    """
    Discover all LoRA files in a specific subfolder.

    Args:
        subfolder: Subfolder name (empty string for root)
        base_path: Base path to loras directory

    Returns:
        List of LoRA filenames, sorted alphabetically
    """
    loras: List[str] = []

    full_path: str = os.path.join(base_path, subfolder) if subfolder else base_path

    if not os.path.isdir(full_path):
        return loras

    # Common LoRA extensions
    lora_extensions: Tuple[str, ...] = (".safetensors", ".ckpt", ".pt", ".pth")

    try:
        for item in os.listdir(full_path):
            if any(item.lower().endswith(ext) for ext in lora_extensions):
                loras.append(item)
    except (OSError, PermissionError):
        pass

    loras.sort()
    return loras


def clear_discovery_cache() -> None:
    """
    Clear the discovery cache.

    Use this when models/LoRAs have been added or removed.
    """
    global _model_subfolder_cache, _lora_subfolder_cache
    _model_subfolder_cache = None
    _lora_subfolder_cache = None


def resolve_model_path(name: str, subfolder: str) -> str:
    """
    Resolve a checkpoint model file path.

    Args:
        name: Model filename (e.g., 'model.safetensors')
        subfolder: Subfolder within the models directory

    Returns:
        Resolved absolute path to the model file

    Raises:
        ValueError: If name is empty or invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Model name must be a non-empty string, got {name}")

    if subfolder is None:
        subfolder = ""

    # Build path relative to models directory
    if subfolder:
        model_path: str = os.path.join("models", "checkpoints", subfolder, name)
    else:
        model_path = os.path.join("models", "checkpoints", name)

    return model_path


def resolve_lora_path(name: str, subfolder: str) -> str:
    """
    Resolve a LoRA file path.

    Args:
        name: LoRA filename (e.g., 'lora.safetensors')
        subfolder: Subfolder within the loras directory

    Returns:
        Resolved absolute path to the LoRA file

    Raises:
        ValueError: If name is empty or invalid
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"LoRA name must be a non-empty string, got {name}")

    if subfolder is None:
        subfolder = ""

    # Build path relative to loras directory
    if subfolder:
        lora_path: str = os.path.join("models", "loras", subfolder, name)
    else:
        lora_path = os.path.join("models", "loras", name)

    return lora_path


def validate_file_exists(path: str) -> bool:
    """
    Check if a file exists at the given path.

    Args:
        path: File path to check

    Returns:
        True if file exists, False otherwise
    """
    if not path or not isinstance(path, str):
        return False

    return os.path.isfile(path)
