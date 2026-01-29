from collections.abc import Iterable

import torch
from torch import ByteTensor

import torch_openreg._C  # type: ignore[misc]
from . import _lazy_call, _lazy_init, current_device, device_count


__all__ = [
    "get_rng_state",
    "get_rng_state_all",
    "set_rng_state",
    "set_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
]


def get_rng_state(device: int | str | torch.device = "openreg"):
    r"""Return a ByteTensor representing the random number state of the current device.

    Args:
        device (int, str or torch.device): The device to get the RNG state.
            Default: ``'openreg'`` (i.e., ``torch.device('openreg')``, the current OpenReg device).
    """
    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("openreg", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.get_state()


def get_rng_state_all() -> list[ByteTensor]:
    r"""Return a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


# LITERALINCLUDE START: OPENREG GENERATOR PY WRAPPER EXAMPLE
def set_rng_state(new_state: ByteTensor, device: int | str | torch.device = "openreg"):
    r"""Set the random number generator state of the specified device.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (int, str or torch.device): The device to set the RNG state.
            Default: ``'openreg'`` (i.e., ``torch.device('openreg')``, the current OpenReg device).
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("openreg", device)

    def cb():
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = torch_openreg._C._get_default_generator(idx)
        default_generator.set_state(new_state)

    _lazy_call(cb)


# LITERALINCLUDE END: OPENREG GENERATOR PY WRAPPER EXAMPLE


def set_rng_state_all(new_states: Iterable[ByteTensor]) -> None:
    r"""Set the random number generator state of all devices."""
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current backend device.

    It's safe to call this function if OpenReg device is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-devices model, this function will only initialize
        the seed on one device.  To initialize all devices, use :func:`seed_all`.
    """

    def cb():
        idx = current_device()
        default_generator = torch_openreg._C._get_default_generator(idx)
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number for all backend devices.

    It's safe to call this function if OpenReg device is not available; in that case, it is silently ignored.
    """

    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch_openreg._C._get_default_generator(i)
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


def initial_seed() -> int:
    r"""Returns the initial seed for generating random numbers for the current backend device.

    .. warning::
        This function eagerly initializes OpenReg device.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch_openreg._C._get_default_generator(idx)
    return default_generator.initial_seed()


# LITERALINCLUDE START: OPENREG MANUAL SEED
def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current backend device.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = torch_openreg._C._get_default_generator(idx)
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


# LITERALINCLUDE END: OPENREG MANUAL SEED


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all backend devices.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for idx in range(device_count()):
            default_generator = torch_openreg._C._get_default_generator(idx)
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)
