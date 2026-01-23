from __future__ import annotations

from ..common import DeviceOpOverrides, register_device_op_overrides


class MTIADeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch._C import _mtia_getCurrentRawStream as {name}"

    def set_device(self, device_idx: int) -> str:
        return f"torch.mtia.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.mtia.synchronize()"

    def device_guard(self, device_idx: int) -> str:
        return f"torch.mtia.device({device_idx})"


register_device_op_overrides("mtia", MTIADeviceOpOverrides())
