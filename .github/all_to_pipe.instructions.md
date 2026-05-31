---
name: all_to_pipe module
description: "ModifyQuality assurance for all_to_pipe: strict typing, module boundaries, unit tests. Use when: working on any file in alltopipe_types/, common/, or nodes/ folders."
applyTo: "alltopipe_types/**,common/**,nodes/**"
---

# all_to_pipe Module Guidelines

## Project Overview
**all_to_pipe** builds reusable, strongly-typed generation pipelines for ComfyUI. It encapsulates models, LoRAs, parameters, prompts, and image settings into explicit `Pipe` objects with no defaults—every value is required and validated.

## Folder & File Structure

### `alltopipe_types/` — Core data model (no ComfyUI node logic here)
- **pipe.py** - Central `Pipe` class: container for model, loras, parameters, image_config, prompts, templates
- **model.py** - Model references + `ModelProcessor` (loading, validation)
- **lora.py** - `LoraSpec` (name, subfolder, weights) + `LoraProcessor` (loading, compatibility checking, applying)
- **parameters.py** - `Parameters` class: steps, cfg, sampler, scheduler, denoise, seed
- **image_config.py** - Image dimensions and format settings
- **prompts.py** - `PositivePrompt`/`NegativePrompt` + `PromptProcessor`
- **template.py** - `Template` placeholder replacement + `TemplateParser`
- **tests/** - Unit tests (mirror filename: `test_*.py` for `*.py`)

### `common/` — Shared utilities
- **constants.py** - Sampler/scheduler names, validation ranges
- **validators.py** - Input validation functions
- **utils.py** - General helper functions
- **prompt_helpers.py** - Prompt text processing
- **file_helpers.py** - File I/O operations
- **companion_loader.py** - Load companion model metadata

### `nodes/` — ComfyUI node implementations
- **update_pipe_node.py** - Merge/update pipe with new data
- **model_node.py, lora_node.py, parameters_builder_node.py, etc.** - Component builders
- **export_node.py** - Export for sampler connections
- **export_json_node.py, export_text_node.py** - Export as JSON/text

## Critical Rules

### 🚫 Module Boundaries (Non-negotiable)
- **NEVER modify files outside `custom_nodes/all_to_pipe/`** 
- No patches to parent `comfy/` code
- No modifications to `server.py`, `main.py`, or ComfyUI core

### 🐍 Python Environment
- Always run with venv: `e:/ComfyUI/.venv`
- Activate before any testing/development: `. e:/ComfyUI/.venv/Scripts/Activate.ps1`
- Never use system Python

### ✅ Type Safety & Code Quality
- **Python 3.13+ Strict Typing Required**
  - All code must use modern Python 3.9+ type syntax (available in Python 3.13+)
  - Use built-in types: `list`, `dict`, `tuple`, `set` (NOT `List`, `Dict`, `Tuple`, `Set`)
  - Use union operator: `X | None` or `X | Y` (NOT `Optional[X]` or `Union[X, Y]`)
  - Example conversions:
    - `Optional[str]` → `str | None`
    - `List[int]` → `list[int]`
    - `Dict[str, Any]` → `dict[str, Any]`
    - `Union[int, str]` → `int | str`
  - NO imports from `typing` for deprecated types: Do NOT use `from typing import Dict, List, Tuple, Set, Optional, Union`
  - Only import when necessary: `from typing import Any, TypedDict, TypeVar, etc.` but always try to avoid using those when possible.
- **Always check Pylance errors** before committing
  - Run code through Pylance (Ctrl+Shift+P → "Python: Run Pylance on current file" or check VSCode Problems panel)
  - Fix ALL `type error` and `value error` warnings (blue squiggles)
  - Only deprecation warnings from external libraries (comfy) are acceptable
  - `Any` type only with explicit comment explaining why
- **No unused imports or variables** - remove dead code immediately
- Use `typing` module annotations everywhere (not runtime optionals)

### 🧪 Testing Protocol
**When you modify `folder/filename.py`, you MUST:**
1. Run its test: `folder/tests/test_filename.py`
2. If test doesn't exist, **CREATE IT**
3. All tests must pass before considering work done
4. Run tests: `pytest folder/tests/test_filename.py -v`

**Example:**
- Modify: `alltopipe_types/lora.py` → Run: `alltopipe_types/tests/test_lora.py`
- Modify: `common/validators.py` → Run: `common/tests/test_validators.py` (create if missing)

## Workflow

1. **Code Change**: Edit file, ensure Pylance shows no errors
2. **Run Tests**: Execute test for that file
3. **Fix Failures**: Update code until test passes
4. **Check Imports**: Use Pylance refactoring (Remove Unused Imports)
## Example: Adding a Function

```python
# In alltopipe_types/model.py
from typing import Optional

def validate_model_path(path: str) -> bool:
    """Check if model path is valid."""
    # implementation
```

**Test counterpart** (alltopipe_types/tests/test_model.py):
```python
import pytest
from alltopipe_types.model import validate_model_path

def test_validate_model_path_valid():
    assert validate_model_path("path/to/model.safetensors") is True

def test_validate_model_path_invalid():
    assert validate_model_path("") is False
```

**Then run:**
```bash
cd e:\ComfyUI
. ./.venv/Scripts/Activate.ps1
pytest custom_nodes/all_to_pipe/alltopipe_types/tests/test_model.py -v
```

## Key Design Principles

- **No defaults**: Every value in Pipe must be explicit
- **Strong typing**: Use `TypedDict`, `dataclass`, or class attributes with type hints
- **Validation first**: Check inputs before processing
- **Separation of concerns**: Data model (alltopipe_types) ≠ Nodes (nodes)
