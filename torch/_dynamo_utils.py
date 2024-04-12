import weakref
from typing import Any

"""
Helper utilities for Dynamo that do not require a full import of torch._dynamo.
The motivation for this file is to avoid circular dependencies.
Please do NOT import anything from torch.* into this file, to avoid circular
dependencies.
"""

DISALLOWED_TORCH_FUNCTION_MODES: Any = weakref.WeakSet()


def disallow_torch_function_mode(torch_function_mode):
    """Causes Dynamo to fall back to eager-mode when it sees this TorchFunctionMode."""
    DISALLOWED_TORCH_FUNCTION_MODES.add(torch_function_mode)
