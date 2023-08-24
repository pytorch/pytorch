# Owner(s): ["module: inductor"]

import ctypes
import torch

from torch._inductor.codecache import CUDACodeCache
from torch._inductor.codegen.cuda.cuda_env import nvcc_exist
from torch._inductor.exc import CUDACompileError
from torch.testing._internal.common_utils import TestCase as TorchTestCase

_SOURCE_CODE = r"""

#include <stdio.h>

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

class TestCUDACodeCache(TorchTestCase):
    def test_cuda_load(self):
        # Test both .o and .so compilation.
        object_file_path, object_hash_key, source_code_path0 = CUDACodeCache.compile(_SOURCE_CODE, "o")
        dll_wrapper, so_hash_key, source_code_path1 = CUDACodeCache.load(_SOURCE_CODE, "so")
        self.assertNotEqual(source_code_path0, source_code_path1)
        self.assertNotEqual(object_hash_key, so_hash_key)

        # Test load and call functions in .so.
        x = torch.rand(10).float().cuda()
        y = torch.rand(10).float().cuda()
        a = 5.0
        expected_y = a * x + y
        res = dll_wrapper.saxpy(
            ctypes.c_int(10),
            ctypes.c_float(a),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
        )
        torch.testing.assert_close(y, expected_y)


    def test_compilation_error(self):
        error_source_code = _SOURCE_CODE.replace("saxpy_device", "saxpy_wrong", 1)
        with self.assertRaises(CUDACompileError):
            CUDACodeCache.compile(error_source_code, "o")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if nvcc_exist():
        run_tests("cuda")
