# mypy: allow-untyped-defs
from ..common import DeviceOpOverrides, register_device_op_overrides


class CUDADeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _cuda_getCurrentRawStream as {name}"

    def generate_stream_creation(self, stream_pool):
        stream_creation_str = ""
        for index, num_used in enumerate(stream_pool):
            if num_used > 0:
                stream_creation_str += (
                    f"stream{index}_raw = torch.cuda.Stream()\n"
                )
                stream_creation_str += (
                    f"stream{index} = stream{index}_raw.cuda_stream\n"
                )
        return stream_creation_str

    def set_device(self, device_idx):
        return f"torch.cuda.set_device({device_idx})"

    def synchronize(self):
        return "torch.cuda.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.cuda._DeviceGuard({device_idx})"


register_device_op_overrides("cuda", CUDADeviceOpOverrides())
