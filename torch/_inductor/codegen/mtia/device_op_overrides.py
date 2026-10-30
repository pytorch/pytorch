from __future__ import annotations

from ..common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class MTIADeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch._C import _mtia_getCurrentRawStream as {name}"

    def set_device(self, device_idx: DeviceIdx) -> str:
        return f"torch.mtia.set_device({device_idx})"

    def synchronize(self) -> str:
        return "torch.mtia.synchronize()"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        return f"torch.mtia.device({device_idx})"

    def current_stream(self) -> str:
        return "torch.mtia.current_stream()"

    def aten_device_type(self) -> str:
        return "at::kMTIA"

    # The C++ types below are defined by the out-of-tree MTIA backend.
    def cpp_aoti_device_guard(self) -> str:
        return "MTIAInductorDeviceGuard"

    def cpp_aoti_stream_guard(self) -> str:
        return "MTIAInductorStreamGuard"

    def cpp_kernel_type(self) -> str:
        return "TorchInductorMTIAKernel*"


register_device_op_overrides("mtia", MTIADeviceOpOverrides())
