import functools
import torch
import importlib.util

def _check_module_exists(name: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except ImportError:
        return False

@functools.lru_cache()
def dill_available():
    return (
        _check_module_exists("dill")
        # dill fails to import under torchdeploy
        and not torch._running_with_deploy()
    )
