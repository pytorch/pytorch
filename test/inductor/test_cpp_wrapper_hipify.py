# Owner(s): ["module: inductor"]
import torch
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.codegen.common import get_device_op_overrides
from torch._inductor.test_case import run_tests, TestCase


TEST_CODES = [
    "CUresult code = EXPR;",
    "CUfunction kernel = nullptr;",
    "static CUfunction kernel = nullptr;",
    "CUdeviceptr var = reinterpret_cast<CUdeviceptr>(arg.data_ptr());",
    "at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal());",
    # Hipification should be idempotent, hipifying should be a no-op for already hipified files
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
]

HIP_CODES = [
    "hipError_t code = EXPR;",
    "hipFunction_t kernel = nullptr;",
    "static hipFunction_t kernel = nullptr;",
    "hipDeviceptr_t var = reinterpret_cast<hipDeviceptr_t>(arg.data_ptr());",
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
    "at::hip::HIPStreamGuardMasqueradingAsCUDA guard(at::hip::getStreamFromExternalMasqueradingAsCUDA());",
]


class TestCppWrapperHipify(TestCase):
    def test_hipify_basic_declaration(self) -> None:
        assert len(TEST_CODES) == len(HIP_CODES)
        for i in range(len(TEST_CODES)):
            result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            expected = HIP_CODES[i]
            self.assertEqual(result, expected)

    def test_hipify_aoti_driver_header(self) -> None:
        cuda_codegen = get_device_op_overrides("cuda")
        header = cuda_codegen.kernel_driver()
        expected = """
            #define CUDA_DRIVER_CHECK(EXPR)                    \\
            do {                                               \\
                hipError_t code = EXPR;                          \\
                const char *msg;                               \\
                hipError_t code_get_error = hipDrvGetErrorString(code, &msg); \\
                if (code_get_error != hipSuccess) {          \\
                    throw std::runtime_error(                  \\
                        std::string("CUDA driver error: ") +   \\
                        std::string("invalid error code!"));   \\
                }                                              \\
                if (code != hipSuccess) {                    \\
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

            static inline hipFunction_t loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                hipModule_t mod;
                hipFunction_t func;
                CUDA_DRIVER_CHECK(hipModuleLoad(&mod, filePath.c_str()));
                CUDA_DRIVER_CHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
                if (sharedMemBytes > 0) {
                    CUDA_DRIVER_CHECK(hipFuncSetAttribute(
                        func,
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        sharedMemBytes
                    ))
                }
                return func;
            }

            static inline void launchKernel(
                    hipFunction_t func,
                    uint32_t gridX,
                    uint32_t gridY,
                    uint32_t gridZ,
                    uint32_t numWarps,
                    uint32_t sharedMemBytes,
                    void* args[],
                    hipStream_t stream) {
                CUDA_DRIVER_CHECK(hipModuleLaunchKernel(
                    func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
                ));
            }
        """
        if torch.version.hip is not None:
            expected = expected.replace("32*numWarps", "64*numWarps")
        result = maybe_hipify_code_wrapper(header, True)
        self.assertEqual(result.rstrip(), expected.rstrip())

    def test_hipify_cross_platform(self) -> None:
        assert len(TEST_CODES) == len(HIP_CODES)
        for i in range(len(TEST_CODES)):
            hip_result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            result = maybe_hipify_code_wrapper(TEST_CODES[i])
            if torch.version.hip is not None:
                self.assertEqual(result, hip_result)
            else:
                self.assertEqual(result, TEST_CODES[i])


if __name__ == "__main__":
    run_tests()
