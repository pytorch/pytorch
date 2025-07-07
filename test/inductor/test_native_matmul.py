# Owner(s): ["module: inductor"]

import os
import sys
import unittest

import torch
from torch import nn
from torch._dynamo.utils import same
from torch._inductor.test_case import run_tests, TestCase, config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    requires_cuda_with_enough_memory,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch._dynamo.testing import rand_strided
from torch._inductor import config as inductor_config


aten = torch.ops.aten

@inductor_config.patch(
    {"triton.enable_native_matmul": True,
     "coordinate_descent_tuning": False}
)
class TestTritonDotReduction(TestCase):
    def test_matmul(self):
        def f(x,y):
            z = x @ y 
            return z 

        M, K, N = 128, 128, 128 
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        compiled = torch.compile(f)
        actual = compiled(x,y)
        expect = f(x,y)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")
        
        code = run_and_get_triton_code(compiled, x,y)
        lines = [line for line in code.split("\n") if "tl.dot" in line]
        assert len(lines) == 1

    @inductor_config.patch({"triton.codegen_upcast_to_fp32": False})
    def test_matmul_fp16(self):
        def f(x,y):
            z = x @ y 
            return z 

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), dtype = torch.float16, device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), dtype = torch.float16, device=GPU_TYPE)

        compiled = torch.compile(f)
        actual = compiled(x,y)
        expect = f(x,y)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")
        
        code = run_and_get_triton_code(compiled, x,y)
        lines = [line for line in code.split("\n") if "tl.dot" in line]
        assert len(lines) == 1

    def test_mm_add(self):
        def f(x,y,z,w):
            return x @ y + z @ w

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        w = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        z = rand_strided((K, N), (N, 1), device=GPU_TYPE)

        compiled = torch.compile(f)
        actual = compiled(x,y,z,w)
        expect = f(x,y,z,w)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")
        
        code = run_and_get_triton_code(compiled, x,y,z,w)
        lines = [line for line in code.split("\n") if "tl.dot" in line]
        assert len(lines) == 2

    def test_mm_complex(self):
        def f(x,y,z,w):
            return x[z] @ y + w + 3 

        M, K, N = 128, 128, 128
        x = rand_strided((M, K), (K, 1), device=GPU_TYPE)
        y = rand_strided((K, N), (N, 1), device=GPU_TYPE)
        
        z = torch.randint(M, (M, K), dtype=torch.long, device=GPU_TYPE)
        w = rand_strided((M, N), (N, 1), device=GPU_TYPE)

        compiled = torch.compile(f)
        actual = compiled(x,y,z,w)
        expect = f(x,y,z,w)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")
        
        code = run_and_get_triton_code(compiled, x,y,z,w)
        lines = [line for line in code.split("\n") if "tl.dot" in line]
        assert len(lines) == 1 


    def test_batchmatmul(self):
        def f(x,y):
            z = torch.bmm(x,y) 
            return z 

        B, M, K, N = 256, 128, 128, 128
        x = rand_strided((B, M, K), (M*K, K, 1), device=GPU_TYPE)
        y = rand_strided((B, K, N), (K*N, N, 1), device=GPU_TYPE)

        compiled = torch.compile(f)
        actual = compiled(x,y)
        expect = f(x,y)
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")
        
        code = run_and_get_triton_code(compiled, x,y)
        lines = [line for line in code.split("\n") if "tl.dot" in line]
        assert len(lines) == 1



if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
