from collections.abc import Iterable

import torch
from torch import Tensor

from ._utils import _device_t, _get_device_index, _lazy_call


def initial_seed() -> int:
    r"""Return the initial seed of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    on the current device (:func:`torch.accelerator.current_device_index`).

    Returns:
        int: the initial seed of the default generator for the current device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = torch.accelerator.current_device_index()
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.initial_seed()


def get_rng_state(device: _device_t = None, /) -> Tensor:
    r"""Return the RNG state of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    as a `torch.Tensor` of dtype `torch.uint8` on the specified accelerator device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the RNG state of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    Returns:
        torch.Tensor: the RNG state of the default generator for the specified device.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = _get_device_index(device, optional=True)
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of `torch.Tensor` of dtype `torch.uint8` representing the RNG states of all devices for
    the current :ref:`accelerator<accelerators>`.

    Returns:
        list[torch.Tensor]: the RNG states of the default generators for all devices.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    return [get_rng_state(i) for i in range(torch.accelerator.device_count())]


def set_rng_state(new_state: Tensor, device: _device_t = None) -> None:
    r"""Set the RNG state of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>` on the given device.

    Args:
        new_state (:class:`torch.Tensor`): The desired RNG state, a tensor of dtype `torch.uint8`.
        device (:class:`torch.device`, str, int, optional): The device to set the RNG state for.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    .. note::
        If the accelerator runtime is not yet initialized, the state is deferred
        and applied once the runtime is ready. See :ref:`lazy-initialization-and-fork-safety-note`.
    """
    if not torch._C._accelerator_isLazyInitialized():
        with torch._C._DisableFuncTorch():
            new_state = new_state.clone(memory_format=torch.contiguous_format)

    device_index = _get_device_index(device) if device is not None else None

    def cb() -> None:
        idx = (
            device_index
            if device_index is not None
            else torch.accelerator.current_device_index()
        )
        default_generator = torch._C._accelerator_getDefaultGenerator(idx)
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    r"""Set the RNG state of the default :class:`torch.Generator` of all devices for the current :ref:`accelerator<accelerators>`.

    Args:
        new_states (Iterable of :class:`torch.Tensor`): The desired RNG states for each device, each tensor of dtype `torch.uint8`.

    .. note::
        If the accelerator runtime is not yet initialized, the state is deferred
        and applied once the runtime is ready. See :ref:`lazy-initialization-and-fork-safety-note`.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers for the current :ref:`accelerator<accelerators>`
    on the current device (:func:`torch.accelerator.current_device_index`).

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-device model, this function is insufficient
        to get determinism. To seed all devices, use :func:`manual_seed_all`.

    .. note::
        If the accelerator runtime is not yet initialized, the state is deferred
        and applied once the runtime is ready. See :ref:`lazy-initialization-and-fork-safety-note`.
    """

    def cb() -> None:
        device_index = torch.accelerator.current_device_index()
        default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Set the seed for generating random numbers on all devices for the current :ref:`accelerator<accelerators>`.

    Args:
        seed (int): The desired seed.

    .. note::
        If the accelerator runtime is not yet initialized, the state is deferred
        and applied once the runtime is ready. See :ref:`lazy-initialization-and-fork-safety-note`.
    """

    def cb() -> None:
        for device_index in range(torch.accelerator.device_count()):
            default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> None:
    r"""Set the seed for generating random numbers to a random number for the current :ref:`accelerator<accelerators>`
    on the current device (:func:`torch.accelerator.current_device_index`).

    .. warning::
        If you are working with a multi-device model, this function is insufficient
        to get determinism. To seed all devices, use :func:`seed_all`.

    .. note::
        If the accelerator runtime is not yet initialized, the state is deferred
        and applied once the runtime is ready. See :ref:`lazy-initialization-and-fork-safety-note`.
    """

    def cb() -> None:
        device_index = torch.accelerator.current_device_index()
        default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
        default_generator.seed()

    _lazy_call(cb)


def seed_all() -> None:
    r"""Set the seed for generating random numbers to a random number on all devices for the current :ref:`accelerator<accelerators>`.

    .. note::
        If the accelerator runtime is not yet initialized, the state is deferred
        and applied once the runtime is ready. See :ref:`lazy-initialization-and-fork-safety-note`.
    """

    def cb() -> None:
        random_seed = 0
        seeded = False
        for i in range(torch.accelerator.device_count()):
            default_generator = torch._C._accelerator_getDefaultGenerator(i)
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)

    _lazy_call(cb)


__all__ = [
    "initial_seed",
    "get_rng_state",
    "get_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "set_rng_state",
    "set_rng_state_all",
]
