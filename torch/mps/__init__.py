r"""
This package enables an interface for accessing MPS (Metal Performance Shaders) backend in Python.
Metal is Apple's API for programming metal GPU (graphics processor unit). Using MPS means that increased
performance can be achieved, by running work on the metal GPU(s).
See https://developer.apple.com/documentation/metalperformanceshaders for more details.
"""
import torch
from .. import Tensor

_is_in_bad_fork = getattr(torch._C, "_mps_is_in_bad_fork", lambda: False)
_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]

# local helper function (not public or exported)
def _get_default_mps_generator() -> torch._C.Generator:
    global _default_mps_generator
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator

def synchronize() -> None:
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    return torch._C._mps_deviceSynchronize()

def get_rng_state() -> Tensor:
    r"""Returns the random number generator state as a ByteTensor."""
    return _get_default_mps_generator().get_state()

def set_rng_state(new_state: Tensor) -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    _get_default_mps_generator().set_state(new_state_copy)

def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    # the torch.mps.manual_seed() can be called from the global
    # torch.manual_seed() in torch/random.py. So we need to make
    # sure mps is available (otherwise we just return without
    # erroring out)
    if not torch._C._has_mps:
        return
    seed = int(seed)
    _get_default_mps_generator().manual_seed(seed)

def seed() -> None:
    r"""Sets the seed for generating random numbers to a random number."""
    _get_default_mps_generator().seed()

def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """
    torch._C._mps_emptyCache()

def set_per_process_memory_fraction(fraction) -> None:
    r"""Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """

    if not isinstance(fraction, float):
        raise TypeError('Invalid type for fraction argument, must be `float`')
    if fraction < 0 or fraction > 2:
        raise ValueError('Invalid fraction value: {}. Allowed range: 0~2'.format(fraction))

    torch._C._mps_setMemoryFraction(fraction)

def current_allocated_memory() -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes.

     .. note::
        The returned size does not include cached allocations in
        memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()

def driver_allocated_memory() -> int:
    r"""Returns total GPU memory allocated by Metal driver for the process in bytes.

     .. note::
        The returned size includes cached allocations in MPSAllocator pools
        as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()

from . import profiler

__all__ = [
    'get_rng_state', 'manual_seed', 'seed', 'set_rng_state', 'synchronize',
    'empty_cache', 'set_per_process_memory_fraction', 'current_allocated_memory',
    'driver_allocated_memory', 'profiler']
