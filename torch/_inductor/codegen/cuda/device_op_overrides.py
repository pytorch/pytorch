# mypy: allow-untyped-defs
from ..common import DeviceOpOverrides, register_device_op_overrides


class CUDADeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch.cuda.set_device({device_idx})"

    def synchronize(self):
        return "torch.cuda.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.cuda._DeviceGuard({device_idx})"

    def cpp_device_guard(self):
        return "at::cuda::CUDAGuard"

    def cpp_aoti_device_guard(self):
        return "AOTICudaGuard"

    def cpp_stream_guard(self):
        return "at::cuda::CUDAStreamGuard"

    def cpp_aoti_stream_guard(self):
        return "AOTICudaStreamGuard"

    def cpp_getStreamFromExternal(self):
        return "at::cuda::getStreamFromExternal"


register_device_op_overrides("cuda", CUDADeviceOpOverrides())
