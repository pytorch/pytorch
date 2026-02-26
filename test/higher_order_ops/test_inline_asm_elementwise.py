# Owner(s): ["module: higher order operators"]
"""
Tests for inline_asm_elementwise higher-order operator.

Tests verify:
1. Bitwise equivalence between eager (Jiterator) and compiled (Inductor) paths
2. Correctness via approximate comparison with reference PyTorch ops
"""
import unittest

import torch
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


# Test case definitions: (name, input_gen_fn, asm_str, constraints, dtype, approx_fn)
def _get_test_cases():
    """Generate test cases as tuples for parametrization."""
    return [
        # Basic float32 operations
        (
            "identity_f32",
            lambda: (torch.randn(100, device="cuda", dtype=torch.float32),),
            "mov.f32 $0, $1;",
            "=f,f",
            torch.float32,
            lambda x: x,
        ),
        (
            "add_f32",
            lambda: (
                torch.randn(100, device="cuda", dtype=torch.float32),
                torch.randn(100, device="cuda", dtype=torch.float32),
            ),
            "add.f32 $0, $1, $2;",
            "=f,f,f",
            torch.float32,
            lambda x, y: x + y,
        ),
        (
            "mul_f32",
            lambda: (
                torch.randn(100, device="cuda", dtype=torch.float32),
                torch.randn(100, device="cuda", dtype=torch.float32),
            ),
            "mul.f32 $0, $1, $2;",
            "=f,f,f",
            torch.float32,
            lambda x, y: x * y,
        ),
        (
            "fma_f32",
            lambda: (
                torch.randn(100, device="cuda", dtype=torch.float32),
                torch.randn(100, device="cuda", dtype=torch.float32),
                torch.randn(100, device="cuda", dtype=torch.float32),
            ),
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            torch.float32,
            lambda a, b, c: a * b + c,
        ),
        # Multi-line PTX
        (
            "double_multiline",
            lambda: (torch.randn(100, device="cuda", dtype=torch.float32),),
            "{.reg .f32 tmp; mov.f32 tmp, $1; add.f32 $0, tmp, tmp;}",
            "=f,f",
            torch.float32,
            lambda x: x * 2,
        ),
        # bf16/fp16 upcasting
        (
            "bf16_upcast",
            lambda: (torch.randn(100, device="cuda", dtype=torch.bfloat16),),
            "add.f32 $0, $1, $1;",
            "=f,f",
            torch.float32,
            lambda x: x.float() * 2,
        ),
        (
            "fp16_upcast",
            lambda: (torch.randn(100, device="cuda", dtype=torch.float16),),
            "add.f32 $0, $1, $1;",
            "=f,f",
            torch.float32,
            lambda x: x.float() * 2,
        ),
        # Integer operations
        (
            "bitwise_and",
            lambda: (
                torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
                torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
            ),
            "and.b32 $0, $1, $2;",
            "=r,r,r",
            torch.int32,
            lambda x, y: x & y,
        ),
        (
            "bitwise_or",
            lambda: (
                torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
                torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
            ),
            "or.b32 $0, $1, $2;",
            "=r,r,r",
            torch.int32,
            lambda x, y: x | y,
        ),
        # Output dtype conversion
        (
            "exponent_extract",
            lambda: (torch.tensor([1.0, 2.0, 0.5, 16.0], device="cuda", dtype=torch.float32),),
            "{.reg .b32 t; mov.b32 t,$1; shr.u32 t,t,23; and.b32 $0,t,0xFF;}",
            "=r,f",
            torch.int32,
            lambda x: ((x.view(torch.int32) >> 23) & 0xFF).to(torch.int32),
        ),
        # Broadcasting
        (
            "broadcast_add",
            lambda: (
                torch.randn(4, 1, device="cuda", dtype=torch.float32),
                torch.randn(1, 8, device="cuda", dtype=torch.float32),
            ),
            "add.f32 $0, $1, $2;",
            "=f,f,f",
            torch.float32,
            lambda x, y: x + y,
        ),
        # Non-contiguous
        (
            "noncontiguous",
            lambda: (torch.randn(8, 16, device="cuda", dtype=torch.float32).t(),),
            "mov.f32 $0, $1;",
            "=f,f",
            torch.float32,
            lambda x: x,
        ),
    ]


TEST_CASES = _get_test_cases()
TEST_CASE_NAMES = [tc[0] for tc in TEST_CASES]


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
@instantiate_parametrized_tests
class TestInlineAsmElementwise(TestCase):
    """Parametrized tests for inline_asm_elementwise."""

    @parametrize("case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i])
    def test_eager_vs_compiled_bitwise(self, case_idx):
        """Verify eager and compiled produce bitwise identical results."""
        name, input_gen_fn, asm_str, constraints, dtype, approx_fn = TEST_CASES[case_idx]

        inputs = input_gen_fn()

        def fn(*args):
            return inline_asm_elementwise(
                *args, asm_str=asm_str, constraints=constraints, dtype=dtype
            )

        eager_result = fn(*inputs)
        compiled_fn = torch.compile(fn, backend="inductor")
        compiled_result = compiled_fn(*inputs)

        self.assertTrue(
            torch.equal(eager_result, compiled_result),
            f"Eager and compiled differ for {name}:\n"
            f"  max diff: {(eager_result.float() - compiled_result.float()).abs().max()}",
        )

    @parametrize("case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i])
    def test_correctness(self, case_idx):
        """Verify result matches reference function."""
        name, input_gen_fn, asm_str, constraints, dtype, approx_fn = TEST_CASES[case_idx]

        inputs = input_gen_fn()

        result = inline_asm_elementwise(
            *inputs, asm_str=asm_str, constraints=constraints, dtype=dtype
        )
        expected = approx_fn(*inputs)

        result_f = result.float() if result.dtype != torch.float32 else result
        expected_f = expected.float() if expected.dtype != torch.float32 else expected

        self.assertTrue(
            torch.allclose(result_f, expected_f, rtol=1e-5, atol=1e-5),
            f"Result differs from expected for {name}:\n"
            f"  max diff: {(result_f - expected_f).abs().max()}",
        )


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestInlineAsmElementwiseErrors(TestCase):
    """Tests for error handling."""

    def test_error_no_inputs(self):
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                asm_str="mov.f32 $0, 1.0;",
                constraints="=f",
                dtype=torch.float32,
            )

    def test_error_constraint_mismatch(self):
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randn(100, device="cuda", dtype=torch.float32)
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                x, y,
                asm_str="add.f32 $0, $1, $2;",
                constraints="=f,f",
                dtype=torch.float32,
            )

    def test_error_cpu_tensor(self):
        x = torch.randn(100, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            inline_asm_elementwise(
                x,
                asm_str="mov.f32 $0, $1;",
                constraints="=f,f",
                dtype=torch.float32,
            )


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
class TestInlineAsmElementwiseEdgeCases(TestCase):
    """Tests for edge cases."""

    def test_empty_tensor(self):
        x = torch.empty(0, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x, asm_str="mov.f32 $0, $1;", constraints="=f,f", dtype=torch.float32
        )
        self.assertEqual(result.shape, torch.Size([0]))

    def test_scalar_tensor(self):
        x = torch.tensor(3.14, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x, asm_str="mov.f32 $0, $1;", constraints="=f,f", dtype=torch.float32
        )
        self.assertEqual(result.shape, torch.Size([]))
        self.assertTrue(torch.allclose(result, x))

    def test_4d_tensor(self):
        x = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x, asm_str="mov.f32 $0, $1;", constraints="=f,f", dtype=torch.float32
        )
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.equal(result, x))

    def test_composition_with_pytorch_ops(self):
        def fn(x, y):
            z = x * 2
            w = inline_asm_elementwise(
                z, y, asm_str="add.f32 $0, $1, $2;", constraints="=f,f,f", dtype=torch.float32
            )
            return w + 1.0

        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randn(100, device="cuda", dtype=torch.float32)

        eager_result = fn(x, y)
        compiled_fn = torch.compile(fn, backend="inductor")
        compiled_result = compiled_fn(x, y)

        self.assertTrue(torch.equal(eager_result, compiled_result))
        self.assertTrue(torch.allclose(eager_result, x * 2 + y + 1.0))

    def test_dynamic_shapes(self):
        def fn(x, y):
            return inline_asm_elementwise(
                x, y, asm_str="add.f32 $0, $1, $2;", constraints="=f,f,f", dtype=torch.float32
            )

        compiled_fn = torch.compile(fn, backend="inductor", dynamic=True)

        for size in [50, 100, 200]:
            x = torch.randn(size, device="cuda", dtype=torch.float32)
            y = torch.randn(size, device="cuda", dtype=torch.float32)
            eager_result = fn(x, y)
            compiled_result = compiled_fn(x, y)
            self.assertTrue(torch.equal(eager_result, compiled_result))


if __name__ == "__main__":
    run_tests()
