# mypy: allow-untyped-defs
import contextlib
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

import torch


__all__ = [
    "set_rng_state",
    "get_rng_state",
    "manual_seed",
    "seed",
    "initial_seed",
    "fork_rng",
    "thread_safe_generator",
]


if TYPE_CHECKING:
    from torch.utils.data._utils.worker import WorkerInfo

from torch._C import default_generator


def set_rng_state(new_state: torch.Tensor) -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.Tensor): The desired state, a tensor of dtype `torch.uint8`

    .. note:: This function only works for CPU. For :ref:`accelerator<accelerators>`, please use
        :func:`torch.accelerator.set_rng_state`.
    """
    default_generator.set_state(new_state)


def get_rng_state() -> torch.Tensor:
    r"""Returns the random number generator state as a `torch.Tensor` of dtype `torch.uint8`.

    See also: :func:`torch.random.fork_rng`.

    .. note:: This function only works for CPU.
        For :ref:`accelerator<accelerators>`, please use :func:`torch.accelerator.get_rng_state`.
    """
    return default_generator.get_state()


def manual_seed(seed: int) -> torch._C.Generator:
    r"""Sets the seed for generating random numbers on all devices. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.

    .. note:: This function seeds both the CPU and the current :ref:`accelerator<accelerators>`.
        For CPU only, use ``torch.default_generator.manual_seed(seed)``.
        For the accelerator only, use :func:`torch.accelerator.manual_seed_all`.
    """
    return _manual_seed_impl(seed)


def _manual_seed_impl(seed: int) -> torch._C.Generator:
    seed = int(seed)
    import torch.accelerator

    if not torch._C._accelerator_isInBadFork():
        torch.accelerator.manual_seed_all(seed)

    return default_generator.manual_seed(seed)


def seed() -> int:
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number on all devices. Returns a 64-bit number used to seed the RNG.

    .. note:: This function generates a non-deterministic seed on CPU, then uses
        it to seed all :ref:`accelerators<accelerators>`.
        For the accelerator only, use :func:`torch.accelerator.manual_seed_all`.
    """
    seed = default_generator.seed()

    import torch.accelerator

    if not torch._C._accelerator_isInBadFork():
        torch.accelerator.manual_seed_all(seed)

    return seed


def initial_seed() -> int:
    r"""Returns the initial seed for generating random numbers as a Python `int`.

    .. note:: The returned seed is for the default generator on CPU only.
        For :ref:`accelerator<accelerators>`, please use :func:`torch.accelerator.initial_seed`.
    """
    return default_generator.initial_seed()


_fork_rng_warned_already = False


@contextlib.contextmanager
def fork_rng(
    devices=None,
    enabled=True,
    _caller="fork_rng",
    _devices_kw="devices",
    device_type: str | None = None,
) -> Generator:
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str | None): device type string, default is ``None``.
            If ``None`` and an accelerator is available, both CPU and
            accelerator RNG states are forked; if no accelerator is available,
            only the CPU RNG state is forked.
            If ``"meta"``, the context manager is a no-op.
            If ``"cpu"``, only the CPU RNG state is forked.
            Otherwise, the value must match the current accelerator or a
            ValueError is raised.
            See :ref:`accelerator<accelerators>` for supported device types.
    """

    if device_type == "meta":
        yield
        return

    if not enabled:
        yield
        return

    acc = torch.accelerator.current_accelerator()
    if device_type == "cpu" or acc is None:
        cpu_rng_state = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)
        return

    if device_type is not None and device_type != acc.type:
        raise ValueError(
            f"Device type '{device_type}' doesn't match the current "
            f"accelerator '{acc.type}'."
        )

    global _fork_rng_warned_already

    if devices is None:
        num_devices = torch.accelerator.device_count()
        if num_devices > 1 and not _fork_rng_warned_already:
            acc_type = acc.type  # pyrefly: ignore [missing-attribute]
            message = (
                f"{acc_type} reports that you have {num_devices} available devices, and "
                f"you have used {_caller} without explicitly specifying which devices are being used. "
                f"For safety, we initialize *every* {acc_type} device by default, which can "
                f"be quite slow if you have a lot of {acc_type}s. If you know that you are only"
                f" making use of a few {acc_type} devices, set the environment variable "
                f"{acc_type}_VISIBLE_DEVICES or the '{_devices_kw}' keyword argument of {_caller} "
                "with the set of devices you are actually using. For example, if you are using CPU only, "
                f"set {acc_type}_VISIBLE_DEVICES= or devices=[]; if you are using device 0 only, "
                f"set {acc_type}_VISIBLE_DEVICES=0 or devices=[0].  To initialize all devices "
                f"and suppress this warning, set the '{_devices_kw}' keyword argument to "
                f"`range(torch.accelerator.device_count())`."
            )
            warnings.warn(message, stacklevel=2)
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = torch.get_rng_state()
    device_rng_states = [torch.accelerator.get_rng_state(device) for device in devices]

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        for device, device_rng_state in zip(devices, device_rng_states):
            torch.accelerator.set_rng_state(device_rng_state, device)


def thread_safe_generator() -> torch.Generator | None:
    """Returns a thread-safe random number generator for use in DataLoader workers.
    This function provides a convenient way for transforms and user code to use
    thread-safe random number generation without manually checking worker context.
    When called in a DataLoader thread worker, returns the worker's thread-local
    :class:`torch.Generator`. When called in the main process or process workers,
    returns ``None`` (which causes PyTorch functions to use the default global RNG).
    Returns:
        Optional[torch.Generator]: Thread-local generator in thread workers, None otherwise.
    Example::
        >>> from torch.random import thread_safe_generator
        >>> generator = thread_safe_generator()
        >>> torch.randint(0, 10, (5,), generator=generator)
    Example with transforms::
        >>> from torch.random import thread_safe_generator
        >>> class MyRandomTransform:
        ...     def __call__(self, img):
        ...         generator = thread_safe_generator()
        ...         offset = torch.randint(0, 10, (2,), generator=generator)
        ...         return img[..., offset[0]:, offset[1]:]
    """
    # Lazy import to avoid circular dependency during torch module initialization
    # torch.__init__ loads torch.random early, but torch.utils.data triggers
    # torch.distributed which needs torch to be fully initialized
    from torch.utils.data import get_worker_info

    worker_info: WorkerInfo | None = get_worker_info()
    if (
        worker_info is not None
        and worker_info.worker_method == "thread"
        and worker_info.rng is not None
    ):
        return worker_info.rng.torch_generator
    return None
