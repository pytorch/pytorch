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
                assert fake.device.type in ("openreg", "cpu"), (  # noqa: S101
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
        torch_openreg._C._set_device(device)

    @staticmethod
    def exchange_device(device):
        return torch_openreg._C._exchangeDevice(device)

    @staticmethod
    def maybe_exchange_device(device):
        if device < 0:
            return -1
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
