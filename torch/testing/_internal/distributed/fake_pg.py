# mypy: allow-untyped-defs

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import FakeProcessGroup, FakeStore


# FakeStore is a no-op Key-Value store (implemented in C++) for initialization
# of the fake process group; one can either use FakeStore or HashStore. It used
# to be a Python class defined here, so it is re-exported to keep
# `from ...fake_pg import FakeStore` working.
__all__ = ["FakeProcessGroup", "FakeStore"]


def _create_fake_pg(common_opts, backend_opts):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convenient tool when playing
    with distributed but don't care about the actual data.
    """
    # new_group() inherits the default backend name but not its options. Keep
    # an explicitly selected fake-world contract consistent across subgroups.
    if backend_opts is None and dist.is_initialized():
        default_pg = dist.distributed_c10d._get_default_group()
        for device in default_pg._device_types:
            default_backend = default_pg._get_backend(device)
            if not isinstance(default_backend, FakeProcessGroup):
                continue
            default_options = default_backend.options
            if default_options is not None and default_options.simulate_uniform_ranks:
                backend_opts = FakeProcessGroup.Options()
                backend_opts.simulate_uniform_ranks = True
                break

    return FakeProcessGroup._create_internal(
        common_opts.group_rank, common_opts.group_size, backend_opts
    )


class _FakeBackendDevices(list):
    """
    Device types supported by the fake backend.

    Third-party accelerators (e.g. npu) register themselves through
    torch.accelerator, but this module may be imported before (or even during)
    the import of the backend module — e.g. the backend autoloads through
    ``torch._import_device_backends`` and its import chain pulls in test infra
    — so the accelerator type cannot be probed eagerly at registration time.
    The device list is consumed lazily when ``init_process_group`` builds its
    device-to-backend map, so resolve the current accelerator's device type
    lazily at read time instead of hardcoding a per-backend whitelist.
    """

    _BASE_DEVICES = ("cpu", "cuda", "hpu", "xpu")

    def __init__(self):
        super().__init__(self._BASE_DEVICES)

    def _accelerator_type(self):
        try:
            acc = torch.accelerator.current_accelerator()
        except Exception:
            # The accelerator module may not be fully initialized yet; fall
            # back to the base device list in that case.
            return None
        return acc.type if acc is not None else None

    def __iter__(self):
        yield from self._BASE_DEVICES
        accelerator_type = self._accelerator_type()
        if accelerator_type is not None and accelerator_type not in self._BASE_DEVICES:
            yield accelerator_type

    def __contains__(self, item):
        return any(device == item for device in self.__iter__())

    def __len__(self):
        accelerator_type = self._accelerator_type()
        has_accelerator = (
            accelerator_type is not None
            and accelerator_type not in self._BASE_DEVICES
        )
        return len(self._BASE_DEVICES) + int(has_accelerator)


dist.Backend.register_backend(
    dist.Backend.FAKE,
    _create_fake_pg,
    extended_api=True,
    devices=_FakeBackendDevices(),
)
