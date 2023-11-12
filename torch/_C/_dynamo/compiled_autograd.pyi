from typing import Callable, Optional

import torch

def set_autograd_compiler(
    autograd_compiler: Optional[
        Callable[[], torch._dynamo.compiled_autograd.AutogradCompilerInstance]
    ]
): ...
def clear_cache(): ...
def is_cache_empty() -> bool: ...
