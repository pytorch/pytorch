from __future__ import annotations

from .common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class MPSDeviceOpOverrides(DeviceOpOverrides):
    # MPS is single-device, so a non-zero index is unexpected. device_idx may be a
    # runtime expr (str) under compile-on-one-rank, but MPS never receives that (it has
    # no current_device_idx_expr override), so guard only the int case.
    def device_guard(self, device_idx: DeviceIdx) -> str:
        if isinstance(device_idx, int) and device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "torch._ops.contextlib.nullcontext()"

    def set_device(self, device_idx: DeviceIdx) -> str:
        if isinstance(device_idx, int) and device_idx != 0:
            raise AssertionError(f"expected device_idx == 0, got {device_idx}")
        return "pass  # MPS set device"

    def kernel_driver(self) -> str:
        return """
            #include <ATen/native/mps/MetalShaderLibrary.h>
        """

    def cpp_kernel_type(self) -> str:
        return "MTLFunction_t"

    def aten_device_type(self) -> str:
        return "at::kMPS"


register_device_op_overrides("mps", MPSDeviceOpOverrides())
