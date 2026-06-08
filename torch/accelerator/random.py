import torch
from torch import Tensor

from ._utils import _device_t, _get_device_index


def initial_seed() -> int:
    r"""Return the initial seed of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    on the current device (:func:`torch.accelerator.current_device_index`).

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = torch.accelerator.current_device_index()
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.initial_seed()


def get_rng_state(device: _device_t = None, /) -> Tensor:
    r"""Return the RNG state of the default :class:`torch.Generator` for the current :ref:`accelerator<accelerators>`
    as a :attr:`torch.uint8` Tensor on the specified accelerator device.

    Args:
        device (:class:`torch.device`, str, int, optional): The device to return the RNG state of.
            If not given, uses :func:`torch.accelerator.current_device_index` by default.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    device_index = _get_device_index(device, optional=True)
    default_generator = torch._C._accelerator_getDefaultGenerator(device_index)
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Return a list of :attr:`torch.uint8` Tensors representing the RNG states of all devices for the current :ref:`accelerator<accelerators>`.

    .. warning::
        This function eagerly initializes the accelerator runtime.
    """
    return [get_rng_state(i) for i in range(torch.accelerator.device_count())]


__all__ = ["initial_seed", "get_rng_state", "get_rng_state_all"]
