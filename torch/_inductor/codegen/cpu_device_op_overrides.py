from __future__ import annotations

from textwrap import dedent

from .common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class NoOpDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def cpp_kernel_type(self) -> str:
        return "void*"

    def set_device(self, device_idx: DeviceIdx) -> str:
        return "pass"

    def synchronize(self) -> str:
        return "pass"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        return "torch._ops.contextlib.nullcontext()"


class CpuDeviceOpOverrides(NoOpDeviceOpOverrides):
    def aten_device_type(self) -> str:
        return "at::kCPU"


register_device_op_overrides("cpu", CpuDeviceOpOverrides())
