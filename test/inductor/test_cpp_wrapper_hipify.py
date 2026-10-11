# Owner(s): ["module: inductor"]
from unittest import mock

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
    "at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal());",
]

HIP_CODES = [
    "hipError_t code = EXPR;",
    "hipFunction_t kernel = nullptr;",
    "static hipFunction_t kernel = nullptr;",
    "hipDeviceptr_t var = reinterpret_cast<hipDeviceptr_t>(arg.data_ptr());",
    "at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal());",
    "at::cuda::CUDAStreamGuard guard(at::cuda::getStreamFromExternal());",
]


class TestCppWrapperHipify(TestCase):
    def test_hipify_basic_declaration(self) -> None:
        if len(TEST_CODES) != len(HIP_CODES):
            raise AssertionError(
                f"TEST_CODES length {len(TEST_CODES)} != HIP_CODES length {len(HIP_CODES)}"
            )
        for i in range(len(TEST_CODES)):
            result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            expected = HIP_CODES[i]
            self.assertEqual(result, expected)

    def test_hipify_aoti_driver_header(self) -> None:
        cuda_codegen = get_device_op_overrides("cuda")
        with mock.patch.object(
            cuda_codegen, "cpp_kernel_launch_supports_pdl", return_value=False
        ):
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

            static inline void setKernelSharedMemory(
                    hipFunction_t func,
                    uint32_t sharedMemBytes) {
                if (sharedMemBytes == 0) {
                    return;
                }
            #if !defined(USE_ROCM)
                // CUDA 13 cuda.h values, spelled out so older toolkits still build:
                // CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE and
                // CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY.
                constexpr int kFuncAttrSharedMemoryMode = 17;
                constexpr int kSharedMemoryModeAllowOversized = 3;
                hipDevice_t device;
                CUDA_DRIVER_CHECK(hipCtxGetDevice(&device));
                int sharedOptin = 0;
                CUDA_DRIVER_CHECK(hipDeviceGetAttribute(
                    &sharedOptin,
                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    device
                ));
                int sharedStatic = 0;
                CUDA_DRIVER_CHECK(hipFuncGetAttribute(
                    &sharedStatic, hipFuncAttributeSharedSizeBytes, func
                ));
                // Above the opt-in limit (sm_107), static + dynamic shared memory
                // is only available in the oversized shared memory mode.
                if (sharedMemBytes + static_cast<uint32_t>(sharedStatic) >
                        static_cast<uint32_t>(sharedOptin)) {
                    CUDA_DRIVER_CHECK(hipFuncSetAttribute(
                        func,
                        static_cast<hipFuncAttribute_t>(kFuncAttrSharedMemoryMode),
                        kSharedMemoryModeAllowOversized
                    ));
                    return;
                }
            #endif
                CUDA_DRIVER_CHECK(hipFuncSetAttribute(
                    func,
                    hipFuncAttributeMaxDynamicSharedMemorySize,
                    sharedMemBytes
                ))
            }

            static inline hipFunction_t loadKernel(
                    std::string filePath,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    const std::optional<std::string> &cubinDir = std::nullopt,
                    std::vector<hipModule_t>* loaded_modules = nullptr) {
                if (cubinDir) {
                    std::filesystem::path p1{*cubinDir};
                    std::filesystem::path p2{filePath};
                    filePath = (p1 / p2.filename()).string();
                }

                hipModule_t mod;
                hipFunction_t func;
                CUDA_DRIVER_CHECK(hipModuleLoad(&mod, filePath.c_str()));
                if (loaded_modules) {
                    loaded_modules->push_back(mod);
                }
                CUDA_DRIVER_CHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
                setKernelSharedMemory(func, sharedMemBytes);
                return func;
            }

            static inline hipFunction_t loadKernel(
                    const void* start,
                    const std::string &funcName,
                    uint32_t sharedMemBytes,
                    std::vector<hipModule_t>* loaded_modules = nullptr) {
                hipModule_t mod;
                hipFunction_t func;
                CUDA_DRIVER_CHECK(hipModuleLoadData(&mod, start));
                if (loaded_modules) {
                    loaded_modules->push_back(mod);
                }
                CUDA_DRIVER_CHECK(hipModuleGetFunction(&func, mod, funcName.c_str()));
                setKernelSharedMemory(func, sharedMemBytes);
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
            # Adjusting the warp size to GPU supported wavefront size on AMD GPU
            prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            expected = expected.replace(
                "32*numWarps", str(prop.warp_size) + "*numWarps"
            )
        result = maybe_hipify_code_wrapper(header, True)
        self.assertEqual(result.rstrip(), expected.rstrip())

    def test_hipify_cross_platform(self) -> None:
        if len(TEST_CODES) != len(HIP_CODES):
            raise AssertionError(
                f"TEST_CODES length {len(TEST_CODES)} != HIP_CODES length {len(HIP_CODES)}"
            )
        for i in range(len(TEST_CODES)):
            hip_result = maybe_hipify_code_wrapper(TEST_CODES[i], True)
            result = maybe_hipify_code_wrapper(TEST_CODES[i])
            if torch.version.hip is not None:
                self.assertEqual(result, hip_result)
            else:
                self.assertEqual(result, TEST_CODES[i])


if __name__ == "__main__":
    run_tests()
