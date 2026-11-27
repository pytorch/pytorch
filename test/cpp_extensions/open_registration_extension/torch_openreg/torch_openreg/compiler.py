import logging

import torch
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.device_interface import (
    DeviceInterface,
    register_interface_for_device,
)

import torch_openreg._C  # type: ignore[misc]


log = logging.getLogger(__name__)


def openreg(gm, example_inputs):
    gm.graph.lint()

    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake = node.meta["val"]
            if isinstance(fake, torch.Tensor):
                if fake.device.type not in ("openreg", "cpu"):
                    raise AssertionError(
                        f"Unexpected device {fake.device} in openreg backend"
                    )

    gm.graph.eliminate_dead_code()
    gm.recompile()

    code = gm.graph.python_code("self")
    log.debug("Compiled graph source:\n%s", code.src)

    return gm.forward


class OpenRegInterface(DeviceInterface):
    class Event:
        def __new__(cls, *args, **kwargs):
            raise NotImplementedError

    class Stream:
        def __new__(cls, *args, **kwargs):
            raise NotImplementedError

    class Worker:
        @staticmethod
        def set_device(device):
            torch_openreg._C._set_device(device)

        @staticmethod
        def current_device():
            return torch_openreg._C._get_device()

        @staticmethod
        def get_device_properties(device=None):
            raise NotImplementedError

    @staticmethod
    def current_device():
        return torch_openreg._C._get_device()

    @staticmethod
    def set_device(device):
        # set_device(-1) must be a no-op: the exchange cleanup hook restores
        # the previous device, which can be -1 when the exchange was skipped.
        if device >= 0:
            torch_openreg._C._set_device(device)

    @staticmethod
    def exchange_device(device):
        # Mirror torch.cuda: a negative index reports no exchange via the -1
        # convention; an out-of-range index raises, like torch.cuda's
        # cudaSetDevice does (and unlike the C++ ExchangeDevice, which would
        # silently set the tracked device to the invalid index).
        if device < 0:
            return -1
        count = torch_openreg._C._get_device_count()
        if device >= count:
            raise RuntimeError(
                "The device index is out of range. It must be in [0, "
                f"{count}), but got {device}."
            )
        return torch_openreg._C._exchangeDevice(device)

    @staticmethod
    def maybe_exchange_device(device):
        # Mirror torch.cuda's _maybe_exchange_device: a negative index reports
        # no exchange via the -1 convention; an out-of-range index is treated
        # as an unusable target and the current device is returned without
        # switching (torch.cuda defers the switch when the target has no
        # primary context). Unlike exchange_device, never raise here: the
        # "maybe" contract treats an unusable target as "no exchange".
        if device < 0:
            return -1
        count = torch_openreg._C._get_device_count()
        if device >= count:
            return torch_openreg._C._get_device()
        return torch_openreg._C._exchangeDevice(device)

    @staticmethod
    def device_count():
        return torch_openreg._C._get_device_count()

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def synchronize(device=None):
        pass

    @staticmethod
    def get_raw_stream(device_idx):
        return 0


register_interface_for_device("openreg", OpenRegInterface)

register_backend(compiler_fn=openreg, name="openreg")
