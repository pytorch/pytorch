# mypy: allow-untyped-defs
from ..common import DeviceOpOverrides, register_device_op_overrides


class XPUDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch._C import _xpu_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch.xpu.set_device({device_idx})"

    def synchronize(self):
        return "torch.xpu.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.xpu._DeviceGuard({device_idx})"

    def cpp_device_guard(self):
        return "at::DeviceGuard"

    def cpp_aoti_device_guard(self):
        return "AOTIXpuGuard"

    def cpp_stream_guard(self):
        return "at::xpu::XPUStreamGuard"

    def cpp_aoti_stream_guard(self):
        return "AOTIXpuStreamGuard"

    def cpp_getStreamFromExternal(self):
        return "at::xpu::getStreamFromExternal"

    def kernel_header(self):
        source_codes = """
        #include <torch/csrc/inductor/aoti_runtime/sycl_runtime_wrappers.h>
        """
        return source_codes

    def kernel_driver(self):
        source_codes = """
            namespace {

            struct Grid {
                Grid(uint32_t x, uint32_t y, uint32_t z)
                  : grid_x(x), grid_y(y), grid_z(z) {}
                uint32_t grid_x;
                uint32_t grid_y;
                uint32_t grid_z;

                bool is_non_zero() {
                    return grid_x > 0 && grid_y > 0 && grid_z > 0;
                }
            };

            }  // anonymous namespace

        """
        return source_codes

    def abi_compatible_header(self):
        return """
        #include <torch/csrc/inductor/aoti_runtime/utils_xpu.h>
        #include <torch/csrc/inductor/aoti_runtime/sycl_runtime_wrappers.h>
        """

    def cpp_stream_type(self):
        return "sycl::queue*"

    def aoti_get_stream(self):
        return "aoti_torch_get_current_xpu_stream"

    def cpp_kernel_type(self):
        return "std::unique_ptr<sycl::kernel>"

    def cpp_device_ptr(self):
        return "void *"


register_device_op_overrides("xpu", XPUDeviceOpOverrides())
