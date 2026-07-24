# Owner(s): ["module: inductor"]

import ctypes
import shutil
import unittest

import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import CUDACodeCache, ROCmCodeCache
from torch._inductor.codegen.cuda.compile_utils import _cuda_compiler
from torch._inductor.codegen.cuda.cuda_env import nvcc_exist
from torch._inductor.codegen.rocm.compile_command import rocm_compiler
from torch._inductor.exc import CUDACompileError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _has_rocm_compiler():
    compiler = rocm_compiler()
    return compiler is not None and shutil.which(compiler) is not None


def _has_gpu_codecache_compiler():
    return _has_rocm_compiler() if TEST_WITH_ROCM else nvcc_exist(_cuda_compiler())


def _gpu_codecache():
    return ROCmCodeCache if TEST_WITH_ROCM else CUDACodeCache


_SOURCE_CODE = r"""

#include <stdio.h>

#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_runtime.h>
#endif

__global__
void saxpy_device(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

extern "C" {

__attribute__((__visibility__("default")))
int saxpy(int n, float a, float *x, float *y) {
  // Perform SAXPY
  saxpy_device<<<(n+255)/256, 256>>>(n, a, x, y);
  return 0;
}

}
"""


@unittest.skipUnless(_has_gpu_codecache_compiler(), "requires nvcc or ROCm compiler")
class TestCUDACodeCache(InductorTestCase):
    @requires_cuda_and_triton
    def test_cuda_load(self):
        with fresh_cache():
            codecache = _gpu_codecache()
            # Test both .o and .so compilation.
            (
                object_file_path,
                object_hash_key,
                source_code_path0,
            ) = codecache.compile(_SOURCE_CODE, "o")
            dll_wrapper, so_hash_key, source_code_path1 = codecache.load(
                _SOURCE_CODE, "so"
            )
            if TEST_WITH_ROCM:
                self.assertTrue(object_file_path.endswith(".o"))
                self.assertTrue(source_code_path0.endswith(".cpp"))
                self.assertTrue(source_code_path1.endswith(".cpp"))
            else:
                self.assertEqual(source_code_path0, source_code_path1)
                self.assertEqual(object_hash_key, so_hash_key)

            # Test load and call functions in .so.
            x = torch.rand(10).float().cuda()
            y = torch.rand(10).float().cuda()
            a = 5.0
            expected_y = a * x + y
            dll_wrapper.saxpy(
                ctypes.c_int(10),
                ctypes.c_float(a),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
            )
            torch.testing.assert_close(y, expected_y)

    @requires_cuda_and_triton
    def test_compilation_error(self):
        with fresh_cache():
            codecache = _gpu_codecache()
            error_source_code = _SOURCE_CODE.replace("saxpy_device", "saxpy_wrong", 1)
            with self.assertRaises(CUDACompileError):
                codecache.compile(error_source_code, "o")

    @requires_cuda_and_triton
    def test_async_compile(self):
        with fresh_cache():
            async_compile = AsyncCompile()
            if TEST_WITH_ROCM:
                compiled_res = async_compile.rocm(_SOURCE_CODE, "so")
            else:
                compiled_res = async_compile.cuda(_SOURCE_CODE, "so")
            async_compile.wait(globals())

            # Test load and call functions in .so.
            x = torch.rand(5).float().cuda()
            y = torch.rand(5).float().cuda()
            a = 2.0
            expected_y = a * x + y
            compiled_res.result().saxpy(
                ctypes.c_int(5),
                ctypes.c_float(a),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
            )
            torch.testing.assert_close(y, expected_y)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests("cuda")
