from ..common import DeviceOverrides

class CUDADeviceOverrides(DeviceOverrides):
    @classmethod
    def py_import_get_raw_stream_as(self, name):
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    @classmethod
    def py_set_device(self, device_idx):
        return f"torch.cuda.set_device({device_idx})"

    @classmethod
    def py_synchronize(self):
        return "torch.cuda.synchronize()"

    @classmethod
    def py_DeviceGuard(self, device_idx):
        return f"torch.cuda._DeviceGuard({device_idx})"

    @classmethod
    def cpp_defAOTStreamGuard(self, name, stream, device_idx):
        return f"AOTICudaStreamGuard {name}({stream}, {device_idx});"

    @classmethod
    def cpp_defStreamGuard(self, name, stream):
        return f"at::cuda::CUDAStreamGuard {name}({stream});"

    @classmethod
    def cpp_getStreamFromExternal(self, stream, device_idx):
        return f"at::cuda::getStreamFromExternal({stream}, {device_idx})"

    @classmethod
    def cpp_defGuard(self, name, device_idx):
        return f"at::cuda::CUDAGuard {name}({device_idx});"

