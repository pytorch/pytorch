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
        with (
            mock.patch.object(torch.version, "hip", "test-hip"),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
        ):
            header = cuda_codegen.kernel_driver()

        self.assertIn("hipModuleLaunchKernel", header)
        self.assertIn("bool launch_pdl = false", header)
        self.assertIn("PDL launch is not supported on HIP", header)
        self.assertNotIn("CUlaunchAttribute", header)
        self.assertNotIn("CUlaunchConfig", header)
        self.assertNotIn("cuLaunchKernelEx", header)

        result = maybe_hipify_code_wrapper(header, True)
        self.assertIn("hipModuleLaunchKernel", result)
        self.assertNotIn("CUlaunchAttribute", result)
        self.assertNotIn("CUlaunchConfig", result)
        self.assertNotIn("cuLaunchKernelEx", result)

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
