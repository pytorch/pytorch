from __future__ import annotations

from .common import DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    def device_guard(self, device_idx: int) -> str:
        assert device_idx == 0
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: int) -> str:
        assert device_idx == 0
        return "# MPS set device"


register_device_op_overrides("mps", MPSDeviceOpOverrides())
