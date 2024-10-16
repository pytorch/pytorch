# mypy: allow-untyped-defs
import torch

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

    def kernel_header(self):
        source_codes = """
        #include <c10/cuda/CUDAGuard.h>
        #include <c10/cuda/CUDAStream.h>
        #include <ATen/cuda/EmptyTensor.h>
        """
        return source_codes

    def kernel_driver(self):
        source_codes = """
            #define CUDA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                CUresult code = EXPR;                          \\
                const char *msg;                               \\
                CUresult code_get_error = cuGetErrorString(code, &msg); \\
                if (code_get_error != CUDA_SUCCESS) {          \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string("invalid error code!"));   \\
                }                                              \\
                if (code != CUDA_SUCCESS) {                    \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string(msg));                     \\
                }                                              \\
            } while (0);

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

            static inline CUfunction loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                CUmodule mod;
                CUfunction func;
                CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
                CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(cuFuncSetAttribute(
                        func,
                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline void launchKernel(
                    CUfunction func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    cudaStream_t stream) {
                CUDA_DRIVER_CHECK(cuLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
        """
        if torch.version.hip is not None:
            # Adjusting the warp size to GPU supported wavefront size on AMD GPU
            prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            source_codes = source_codes.replace(
                "32*numWarps", str(prop.warp_size) + "*numWarps"
            )
        return source_codes

    def abi_compatible_header(self):
        return "#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>"

    def cpp_stream_type(self):
        return "cudaStream_t"

    def aoti_get_stream(self):
        return "aoti_torch_get_current_cuda_stream"

    def cpp_kernel_type(self):
        return "CUfunction"

    def cpp_device_ptr(self):
        return "CUdeviceptr"


register_device_op_overrides("cuda", CUDADeviceOpOverrides())
