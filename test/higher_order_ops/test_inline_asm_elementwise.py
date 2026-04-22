# Owner(s): ["module: higher order operators"]
"""
Tests for inline_asm_elementwise higher-order operator.

Tests verify:
1. Bitwise equivalence between eager (Jiterator) and compiled (Inductor) paths
2. Correctness via approximate comparison with reference PyTorch ops
"""

import unittest
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch.testing._internal.common_cuda import evaluate_gfx_arch_within, SM70OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MI200_ARCH,
    MI300_ARCH,
    NAVI_ARCH,
    parametrize,
    run_tests,
    TEST_CUDA,
    TestCase,
)


@dataclass
class AsmTestCase:
    name: str
    input_gen_fn: Callable
    asm_str: str
    constraints: str
    dtype: torch.dtype
    approx_fn: Callable
    pack: int = 1
    compile_only: bool = False
    min_sm: int = 70


TEST_CASES = [
    # Basic float32 operations
    AsmTestCase(
        "identity_f32",
        lambda: (torch.randn(100, device="cuda", dtype=torch.float32),),
        "v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x,
    ),
    AsmTestCase(
        "add_f32",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
        ),
        "v_add_f32 $0, $1, $2" if torch.version.hip else "add.f32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=f,f,f",
        torch.float32,
        lambda x, y: x + y,
    ),
    AsmTestCase(
        "mul_f32",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
        ),
        "v_mul_f32 $0, $1, $2" if torch.version.hip else "mul.f32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=f,f,f",
        torch.float32,
        lambda x, y: x * y,
    ),
    AsmTestCase(
        "fma_f32",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
            torch.randn(100, device="cuda", dtype=torch.float32),
        ),
        "v_fma_f32 $0, $1, $2, $3"
        if torch.version.hip
        else "fma.rn.f32 $0, $1, $2, $3;",
        "=v, v, v, v" if torch.version.hip else "=f,f,f,f",
        torch.float32,
        lambda a, b, c: a * b + c,
    ),
    # Multi-line inline asm. PTX uses curly braces; AMDGCN uses newlines.
    AsmTestCase(
        "double_multiline",
        lambda: (torch.randn(100, device="cuda", dtype=torch.float32),),
        (
            """
            v_mov_b32 $0, $1
            v_add_f32 $0, $0, $1
            """
            if torch.version.hip
            else "{.reg .f32 tmp; mov.f32 tmp, $1; add.f32 $0, tmp, tmp;}"
        ),
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x * 2,
    ),
    # bf16/fp16 upcasting (compile-only: Jiterator can't handle dtype mismatch)
    AsmTestCase(
        "bf16_upcast",
        lambda: (torch.randn(100, device="cuda", dtype=torch.bfloat16),),
        "v_add_f32 $0, $1, $1" if torch.version.hip else "add.f32 $0, $1, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x.float() * 2,
        compile_only=True,
        min_sm=80,
    ),
    AsmTestCase(
        "fp16_upcast",
        lambda: (torch.randn(100, device="cuda", dtype=torch.float16),),
        "v_add_f32 $0, $1, $1" if torch.version.hip else "add.f32 $0, $1, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x.float() * 2,
        compile_only=True,
    ),
    # Integer operations
    AsmTestCase(
        "bitwise_and",
        lambda: (
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
        ),
        "v_and_b32 $0, $1, $2" if torch.version.hip else "and.b32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=r,r,r",
        torch.int32,
        lambda x, y: x & y,
    ),
    AsmTestCase(
        "bitwise_or",
        lambda: (
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
            torch.randint(0, 2**16, (100,), device="cuda", dtype=torch.int32),
        ),
        "v_or_b32 $0, $1, $2" if torch.version.hip else "or.b32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=r,r,r",
        torch.int32,
        lambda x, y: x | y,
    ),
    # Output dtype differs from input (compile-only: Jiterator returns input dtype)
    # AMDGCN: v_bfe_u32 (bit-field extract) replaces PTX's multi-instruction
    # shift-and-mask sequence in a single instruction.
    AsmTestCase(
        "exponent_extract",
        lambda: (
            torch.tensor([1.0, 2.0, 0.5, 16.0], device="cuda", dtype=torch.float32),
        ),
        (
            "v_bfe_u32 $0, $1, 23, 8"
            if torch.version.hip
            else "{.reg .b32 t; mov.b32 t,$1; shr.u32 t,t,23; and.b32 $0,t,0xFF;}"
        ),
        "=v, v" if torch.version.hip else "=r,f",
        torch.int32,
        lambda x: ((x.view(torch.int32) >> 23) & 0xFF).to(torch.int32),
        compile_only=True,
    ),
    # Truncate u32 -> u16 (compile-only).
    # PTX: uses "h" (16-bit) output / "r" (32-bit) input constraints.
    # AMDGCN: VGPRs are always 32-bit (no "h" equivalent), so we use "v"
    # and extract the lower 16 bits via v_bfe_u32.
    AsmTestCase(
        "truncate_to_uint16",
        lambda: (torch.randint(0, 256, (100,), device="cuda", dtype=torch.int32),),
        "v_bfe_u32 $0, $1, 0, 16" if torch.version.hip else "cvt.u16.u32 $0, $1;",
        "=v, v" if torch.version.hip else "=h,r",
        torch.uint16,
        lambda x: x.to(torch.uint16),
        compile_only=True,
    ),
    # Broadcasting
    AsmTestCase(
        "broadcast_add",
        lambda: (
            torch.randn(4, 1, device="cuda", dtype=torch.float32),
            torch.randn(1, 8, device="cuda", dtype=torch.float32),
        ),
        "v_add_f32 $0, $1, $2" if torch.version.hip else "add.f32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=f,f,f",
        torch.float32,
        lambda x, y: x + y,
    ),
    # Non-contiguous
    AsmTestCase(
        "noncontiguous",
        lambda: (torch.randn(8, 16, device="cuda", dtype=torch.float32).t(),),
        "v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x,
    ),
    # fp16/bf16 native asm (compile-only: inductor computes in fp32, needs downcast)
    # ROCm: Inductor feeds f32 values (upcasted for computation).  AMDGCN has no
    # "h" constraint for 16-bit regs, so we add in f32 and convert to the target
    # format.  PTX "h" constraints tell Triton to downcast before the asm.
    AsmTestCase(
        "add_fp16_native",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.float16),
            torch.randn(100, device="cuda", dtype=torch.float16),
        ),
        (
            "v_add_f32 $0, $1, $2\nv_cvt_f16_f32 $0, $0"
            if torch.version.hip
            else "add.f16 $0, $1, $2;"
        ),
        "=v,v,v" if torch.version.hip else "=h,h,h",
        torch.float16,
        lambda x, y: x + y,
        compile_only=True,
    ),
    # AMDGCN: v_cvt_pk_bf16_f32 packs two f32 values into bf16 in a single
    # 32-bit register.  We pass $0 twice — only the lower 16 bits (first
    # bf16 slot) are used by Triton.
    AsmTestCase(
        "add_bf16_native",
        lambda: (
            torch.randn(100, device="cuda", dtype=torch.bfloat16),
            torch.randn(100, device="cuda", dtype=torch.bfloat16),
        ),
        (
            "v_add_f32 $0, $1, $2\nv_cvt_pk_bf16_f32 $0, $0, $0"
            if torch.version.hip
            else "add.bf16 $0, $1, $2;"
        ),
        "=v,v,v" if torch.version.hip else "=h,h,h",
        torch.bfloat16,
        lambda x, y: x + y,
        compile_only=True,
        min_sm=90,
    ),
    # pack=2: each asm invocation processes 2 elements (compile-only)
    AsmTestCase(
        "identity_pack2",
        lambda: (torch.randn(128, device="cuda", dtype=torch.float32),),
        (
            """
            v_mov_b32 $0, $2
            v_mov_b32 $1, $3
            """
            if torch.version.hip
            else "mov.b32 $0, $2; mov.b32 $1, $3;"
        ),
        "=v,=v,v,v" if torch.version.hip else "=r,=r,r,r",
        torch.float32,
        lambda x: x,
        pack=2,
        compile_only=True,
    ),
    AsmTestCase(
        "add_pack2",
        lambda: (
            torch.randn(128, device="cuda", dtype=torch.float32),
            torch.randn(128, device="cuda", dtype=torch.float32),
        ),
        (
            """
            v_add_f32 $0, $2, $4
            v_add_f32 $1, $3, $5
            """
            if torch.version.hip
            else "add.f32 $0, $2, $4; add.f32 $1, $3, $5;"
        ),
        "=v,=v,v,v,v,v" if torch.version.hip else "=f,=f,f,f,f,f",
        torch.float32,
        lambda x, y: x + y,
        pack=2,
        compile_only=True,
    ),
]
TEST_CASE_NAMES = [tc.name for tc in TEST_CASES]


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
@instantiate_parametrized_tests
class TestInlineAsmElementwise(TestCase):
    """Parametrized tests for inline_asm_elementwise."""

    @parametrize(
        "case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i]
    )
    def test_eager_vs_compiled_bitwise(self, case_idx):
        """Verify eager and compiled produce bitwise identical results."""
        tc = TEST_CASES[case_idx]
        if not torch.version.hip and torch.cuda.get_device_capability() < (
            tc.min_sm // 10,
            tc.min_sm % 10,
        ):
            self.skipTest(f"Requires SM{tc.min_sm}+")

        # Native bf16 conversion instruction not available before gfx950.
        if (
            torch.version.hip
            and tc.name == "add_bf16_native"
            and evaluate_gfx_arch_within(
                [
                    *MI200_ARCH,
                    *MI300_ARCH,
                    *NAVI_ARCH,
                ]
            )
        ):
            self.skipTest("Requires gfx950+")

        inputs = tc.input_gen_fn()

        def fn(*args):
            return inline_asm_elementwise(
                *args,
                asm_str=tc.asm_str,
                constraints=tc.constraints,
                dtype=tc.dtype,
                pack=tc.pack,
            )

        torch._dynamo.reset()
        compiled_result = torch.compile(fn, backend="inductor")(*inputs)

        if tc.compile_only:
            expected = tc.approx_fn(*inputs)
            self.assertEqual(
                compiled_result.float(), expected.float(), atol=1e-5, rtol=1e-5
            )
        else:
            eager_result = fn(*inputs)
            self.assertEqual(eager_result, compiled_result)

    @parametrize(
        "case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i]
    )
    def test_correctness(self, case_idx):
        """Verify result matches reference function."""
        tc = TEST_CASES[case_idx]
        if not torch.version.hip and torch.cuda.get_device_capability() < (
            tc.min_sm // 10,
            tc.min_sm % 10,
        ):
            self.skipTest(f"Requires SM{tc.min_sm}+")

        # Native bf16 conversion instruction not available before gfx950.
        if (
            torch.version.hip
            and tc.name == "add_bf16_native"
            and evaluate_gfx_arch_within(
                [
                    *MI200_ARCH,
                    *MI300_ARCH,
                    *NAVI_ARCH,
                ]
            )
        ):
            self.skipTest("Requires gfx950+")

        inputs = tc.input_gen_fn()

        def fn(*args):
            return inline_asm_elementwise(
                *args,
                asm_str=tc.asm_str,
                constraints=tc.constraints,
                dtype=tc.dtype,
                pack=tc.pack,
            )

        if tc.compile_only:
            torch._dynamo.reset()
            result = torch.compile(fn, backend="inductor")(*inputs)
        else:
            result = fn(*inputs)
        expected = tc.approx_fn(*inputs)

        self.assertEqual(result.float(), expected.float(), atol=1e-5, rtol=1e-5)


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
class TestInlineAsmElementwiseErrors(TestCase):
    """Tests for error handling."""

    def test_error_no_inputs(self):
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                asm_str="v_mov_b32 $0, 1.0"
                if torch.version.hip
                else "mov.f32 $0, 1.0;",
                constraints="=v" if torch.version.hip else "=f",
                dtype=torch.float32,
            )

    def test_error_constraint_mismatch(self):
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randn(100, device="cuda", dtype=torch.float32)
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                x,
                y,
                asm_str="v_add_f32 $0, $1, $2"
                if torch.version.hip
                else "add.f32 $0, $1, $2;",
                constraints="=v,v" if torch.version.hip else "=f,f",
                dtype=torch.float32,
            )

    def test_error_mixed_dtypes(self):
        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randint(0, 10, (100,), device="cuda", dtype=torch.int32)
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                x,
                y,
                asm_str="v_add_f32 $0, $1, $2"
                if torch.version.hip
                else "add.f32 $0, $1, $2;",
                constraints="=v,v,v" if torch.version.hip else "=f,f,r",
                dtype=torch.float32,
            )

    def test_error_cpu_tensor(self):
        x = torch.randn(100, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            inline_asm_elementwise(
                x,
                asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
                constraints="=v,v" if torch.version.hip else "=f,f",
                dtype=torch.float32,
            )


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
class TestInlineAsmElementwiseEdgeCases(TestCase):
    """Tests for edge cases."""

    def test_empty_tensor(self):
        x = torch.empty(0, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x,
            asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
            constraints="=v, v" if torch.version.hip else "=f,f",
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, torch.Size([0]))

    def test_scalar_tensor(self):
        x = torch.tensor(3.14, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x,
            asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
            constraints="=v, v" if torch.version.hip else "=f,f",
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, torch.Size([]))
        self.assertEqual(result, x)

    def test_4d_tensor(self):
        x = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float32)
        result = inline_asm_elementwise(
            x,
            asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
            constraints="=v, v" if torch.version.hip else "=f,f",
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result, x)

    def test_composition_with_pytorch_ops(self):
        def fn(x, y):
            z = x * 2
            w = inline_asm_elementwise(
                z,
                y,
                asm_str="v_add_f32 $0, $1, $2"
                if torch.version.hip
                else "add.f32 $0, $1, $2;",
                constraints="=v, v, v" if torch.version.hip else "=f,f,f",
                dtype=torch.float32,
            )
            return w + 1.0

        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.randn(100, device="cuda", dtype=torch.float32)

        eager_result = fn(x, y)
        compiled_fn = torch.compile(fn, backend="inductor")
        compiled_result = compiled_fn(x, y)

        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(eager_result, x * 2 + y + 1.0)

    def test_output_strides_mixed_inputs(self):
        """Verify fake mode output strides match eager (TensorIterator) strides."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Two inputs with different strides: one contiguous, one transposed.
        # This exercises TensorIterator's slow path for stride computation.
        x = torch.randn(8, 16, device="cuda", dtype=torch.float32)
        y = torch.randn(16, 8, device="cuda", dtype=torch.float32).t()

        eager_result = inline_asm_elementwise(
            x,
            y,
            asm_str="v_add_f32 $0, $1, $2"
            if torch.version.hip
            else "add.f32 $0, $1, $2;",
            constraints="=v, v, v" if torch.version.hip else "=f,f,f",
            dtype=torch.float32,
        )

        with FakeTensorMode() as mode:
            fake_x = mode.from_tensor(x)
            fake_y = mode.from_tensor(y)
            fake_result = inline_asm_elementwise(
                fake_x,
                fake_y,
                asm_str="v_add_f32 $0, $1, $2"
                if torch.version.hip
                else "add.f32 $0, $1, $2;",
                constraints="=v, v, v" if torch.version.hip else "=f,f,f",
                dtype=torch.float32,
            )

        self.assertEqual(eager_result.shape, fake_result.shape)
        self.assertEqual(eager_result.stride(), fake_result.stride())

    def test_dynamic_shapes(self):
        def fn(x, y):
            return inline_asm_elementwise(
                x,
                y,
                asm_str="v_add_f32 $0, $1, $2"
                if torch.version.hip
                else "add.f32 $0, $1, $2;",
                constraints="=v, v, v" if torch.version.hip else "=f,f,f",
                dtype=torch.float32,
            )

        compiled_fn = torch.compile(fn, backend="inductor", dynamic=True)

        for size in [50, 100, 200]:
            x = torch.randn(size, device="cuda", dtype=torch.float32)
            y = torch.randn(size, device="cuda", dtype=torch.float32)
            eager_result = fn(x, y)
            compiled_result = compiled_fn(x, y)
            self.assertEqual(eager_result, compiled_result)


@unittest.skipIf(not TEST_CUDA, "CUDA not available")
@unittest.skipIf(not SM70OrLater, "Requires SM70+")
class TestInlineAsmPackPadding(TestCase):
    """Test that pack padding works when block size < pack."""

    def test_pack2_xblock1_padding(self):
        """Force XBLOCK=1 with pack=2 so padding is needed."""
        from torch._inductor.choices import InductorChoices
        from torch._inductor.codegen.triton import FixedTritonConfig
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        class ForceXBlock1(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {
                    **kernel_kwargs,
                    "fixed_config": FixedTritonConfig({"XBLOCK": 1}),
                }

        def fn(x):
            return inline_asm_elementwise(
                x,
                asm_str=(
                    """
                    v_mov_b32 $0, $2
                    v_mov_b32 $1, $3
                    """
                    if torch.version.hip
                    else "mov.b32 $0, $2; mov.b32 $1, $3;"
                ),
                constraints="=v,=v,v,v" if torch.version.hip else "=r,=r,r,r",
                dtype=torch.float32,
                pack=2,
            )

        x = torch.randn(128, device="cuda", dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertEqual(result, x)
        # Verify padding helpers are emitted in the generated code
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)

    def test_pack4_xblock1_padding(self):
        """Force XBLOCK=1 with pack=4 so padding is needed."""
        from torch._inductor.choices import InductorChoices
        from torch._inductor.codegen.triton import FixedTritonConfig
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        class ForceXBlock1(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {
                    **kernel_kwargs,
                    "fixed_config": FixedTritonConfig({"XBLOCK": 1}),
                }

        def fn(x):
            return inline_asm_elementwise(
                x,
                asm_str=(
                    """
                    v_mov_b32 $0, $4
                    v_mov_b32 $1, $5
                    v_mov_b32 $2, $6
                    v_mov_b32 $3, $7
                    """
                    if torch.version.hip
                    else "mov.b32 $0, $4; mov.b32 $1, $5; mov.b32 $2, $6; mov.b32 $3, $7;"
                ),
                constraints=(
                    "=v,=v,=v,=v,v,v,v,v"
                    if torch.version.hip
                    else "=r,=r,=r,=r,r,r,r,r"
                ),
                dtype=torch.float32,
                pack=4,
            )

        x = torch.randn(128, device="cuda", dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertEqual(result, x)
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)

    def test_pack4_xblock2_partial_padding(self):
        """XBLOCK=2 < pack=4, so 1 round of padding is needed (not 2)."""
        from torch._inductor.choices import InductorChoices
        from torch._inductor.codegen.triton import FixedTritonConfig
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        class ForceXBlock2(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {
                    **kernel_kwargs,
                    "fixed_config": FixedTritonConfig({"XBLOCK": 2}),
                }

        def fn(x):
            return inline_asm_elementwise(
                x,
                asm_str=(
                    """
                    v_mov_b32 $0, $4
                    v_mov_b32 $1, $5
                    v_mov_b32 $2, $6
                    v_mov_b32 $3, $7
                    """
                    if torch.version.hip
                    else "mov.b32 $0, $4; mov.b32 $1, $5; mov.b32 $2, $6; mov.b32 $3, $7;"
                ),
                constraints=(
                    "=v,=v,=v,=v,v,v,v,v"
                    if torch.version.hip
                    else "=r,=r,=r,=r,r,r,r,r"
                ),
                dtype=torch.float32,
                pack=4,
            )

        x = torch.randn(128, device="cuda", dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock2()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertEqual(result, x)
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)

    def test_pack2_xblock1_yblock1_padding(self):
        """Force XBLOCK=1, YBLOCK=1 with pack=2 on a 2D-tiled kernel."""
        from torch._inductor.choices import InductorChoices
        from torch._inductor.codegen.triton import FixedTritonConfig
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        class ForceXY1(InductorChoices):
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                return {
                    **kernel_kwargs,
                    "fixed_config": FixedTritonConfig({"XBLOCK": 1, "YBLOCK": 1}),
                }

        def fn(x, y):
            return inline_asm_elementwise(
                x,
                y,
                asm_str=(
                    """
                    v_add_f32 $0, $2, $4
                    v_add_f32 $1, $3, $5
                    """
                    if torch.version.hip
                    else "add.f32 $0, $2, $4; add.f32 $1, $3, $5;"
                ),
                constraints="=v,=v,v,v,v,v" if torch.version.hip else "=f,=f,f,f,f,f",
                dtype=torch.float32,
                pack=2,
            )

        x = torch.randn(8, 16, device="cuda", dtype=torch.float32)
        # Transposed input triggers 2D tiling (different stride patterns)
        y = torch.randn(16, 8, device="cuda", dtype=torch.float32).T
        with torch._inductor.virtualized.V.set_choices_handler(ForceXY1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor"), x, y
            )

        self.assertEqual(result, x + y)
        FileCheck().check("YBLOCK").check("inline_asm_pack").check(
            "inline_asm_unpack"
        ).run(code)


if __name__ == "__main__":
    run_tests()
