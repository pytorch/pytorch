from __future__ import annotations

from .common import NoOpDeviceOpOverrides, register_device_op_overrides


class CpuDeviceOpOverrides(NoOpDeviceOpOverrides):
    def aten_device_type(self) -> str:
        return "at::kCPU"


register_device_op_overrides("cpu", CpuDeviceOpOverrides())
