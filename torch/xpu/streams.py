import ctypes

import torch
from torch._streambase import _StreamBase
from ._utils import _dummy_type


if not hasattr(torch._C, "_XpuStreamBase"):
    # Define dummy base classes
    torch._C.__dict__["_XpuStreamBase"] = _dummy_type("_XpuStreamBase")


class Stream(torch._C._XpuStreamBase, _StreamBase):
    r"""Wrapper around a XPU stream.

    A XPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, should be 0 or
            negative, where negative numbers indicate higher priority. By default,
            streams have priority 0.
    """

    def __new__(cls, device=None, priority=0, **kwargs):
        # setting device manager is expensive, so we avoid it unless necessary
        if device is None or ("stream_id" in kwargs and "device_index" in kwargs):
            return super().__new__(cls, priority=priority, **kwargs)
        else:
            with torch.xpu.device(device):
                return super().__new__(cls, priority=priority, **kwargs)

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def record_event(self, event=None):
        pass

    def query(self):
        r"""Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
        return super().query()

    def synchronize(self):
        r"""Wait for all the kernels in this stream to complete."""
        super().synchronize()

    @property
    def _as_parameter_(self):
        return ctypes.c_void_p(self.sycl_queue)

    def __eq__(self, o):
        if isinstance(o, Stream):
            return super().__eq__(o)
        return False

    def __hash__(self):
        return hash((self.sycl_queue, self.device))

    def __repr__(self):
        return (
            f"<torch.xpu.Stream device={self.device} sycl_queue={self.sycl_queue:#x}>"
        )
