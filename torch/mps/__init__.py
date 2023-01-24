r"""
This unit enables an interface for accessing MPS backend in python
"""
import torch
from functools import lru_cache
import threading
from .. import Tensor
import torch._C

_initialized = False
_initialization_lock = threading.Lock()
default_generator: torch._C.Generator = ()  # type: ignore[assignment]

def init():
    r"""Initialize PyTorch's MPS state.
    Does nothing if the MPS state is already initialized.
    """
    _lazy_init()

def _lazy_init():
    global _initialized
    if is_initialized():
        return
    with _initialization_lock:
        if is_initialized():
            return
        if not hasattr(torch._C, '_mps_init'):
            raise AssertionError("Torch not compiled with MPS enabled")
        torch._C._mps_init()
        _initialized = True

@lru_cache()
def is_available() -> bool:
    r"""Returns a bool indicating if MPS is currently available."""
    return torch._C._is_mps_available()

@lru_cache()
def is_macos13_or_newer() -> bool:
    r"""Returns a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._is_mps_on_macos_13_or_newer()

def synchronize() -> None:
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    _lazy_init()
    return torch._C._mps_synchronize()

def get_rng_state() -> Tensor:
    r"""Returns the random number generator state as a ByteTensor."""
    _lazy_init()
    return default_generator.get_state()

def set_rng_state(new_state: Tensor) -> None:
    r"""Sets the random number generator state.
    Args:
        new_state (torch.ByteTensor): The desired state
    """
    _lazy_init()
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    default_generator.set_state(new_state_copy)

def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers
    Args:
        seed (int): The desired seed.
    """
    if not is_available():
        return
    _lazy_init()
    seed = int(seed)
    default_generator.manual_seed(seed)

def seed() -> None:
    r"""Sets the seed for generating random numbers to a random number."""
    _lazy_init()
    default_generator.seed()

def is_initialized():
    r"""Returns whether PyTorch's MPS state has been initialized."""
    return _initialized

__all__ = [
    'default_generator', 'get_rng_state', 'is_available', 'manual_seed',
    'seed', 'set_rng_state', 'synchronize', 'init', 'is_initialized']
