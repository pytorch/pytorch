from __future__ import annotations

from torch._inductor.codegen.common import DeviceOpOverrides, register_device_op_overrides


class PallasDeviceOpOverrides(DeviceOpOverrides):
    """
    Provides XLA-specific overrides for device operations.
    """

    def device_guard(self, device_idx: int) -> str:
        # This should emit something like:
        # with torch.xla.device(device_idx):
        return ""

    def set_device(self, device_idx: int) -> str:
        # This should emit something like:
        # torch.xla.set_device(device_idx)
        return ""

    def synchronize(self) -> str:
        # This should emit something like:
        # torch.xla.synchronize()
        return ""

register_device_op_overrides("xla", PallasDeviceOpOverrides())