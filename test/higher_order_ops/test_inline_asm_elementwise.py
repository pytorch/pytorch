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
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.exc import TritonUnavailableError
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch.testing._internal.common_cuda import evaluate_gfx_arch_within, SM70OrLater
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    MI200_ARCH,
    MI300_ARCH,
    NAVI3_5_ARCH,
    NAVI3_ARCH,
    NAVI4_ARCH,
    NAVI_ARCH,
    parametrize,
    run_tests,
    skipIfRocm,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.utils._triton import has_triton


# Upstream Triton enables LLVM real-true16 for Navi3, Navi3.5, and Navi4.
TRUE16_ARCH = (*NAVI3_ARCH, *NAVI3_5_ARCH, *NAVI4_ARCH)


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
    true16_asm_str: str | None = None
    true16_constraints: str | None = None


def _get_asm_config(tc: AsmTestCase) -> tuple[str, str]:
    if (
        torch.version.hip
        and (tc.true16_asm_str is not None or tc.true16_constraints is not None)
        and evaluate_gfx_arch_within(TRUE16_ARCH)
    ):
        asm_str = tc.true16_asm_str if tc.true16_asm_str is not None else tc.asm_str
        constraints = (
            tc.true16_constraints
            if tc.true16_constraints is not None
            else tc.constraints
        )
        return asm_str, constraints
    return tc.asm_str, tc.constraints


TEST_CASES = [
    # Basic float32 operations
    AsmTestCase(
        "identity_f32",
        lambda device: (torch.randn(100, device=device, dtype=torch.float32),),
        "v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x,
    ),
    AsmTestCase(
        "add_f32",
        lambda device: (
            torch.randn(100, device=device, dtype=torch.float32),
            torch.randn(100, device=device, dtype=torch.float32),
        ),
        "v_add_f32 $0, $1, $2" if torch.version.hip else "add.f32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=f,f,f",
        torch.float32,
        lambda x, y: x + y,
    ),
    AsmTestCase(
        "mul_f32",
        lambda device: (
            torch.randn(100, device=device, dtype=torch.float32),
            torch.randn(100, device=device, dtype=torch.float32),
        ),
        "v_mul_f32 $0, $1, $2" if torch.version.hip else "mul.f32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=f,f,f",
        torch.float32,
        lambda x, y: x * y,
    ),
    AsmTestCase(
        "fma_f32",
        lambda device: (
            torch.randn(100, device=device, dtype=torch.float32),
            torch.randn(100, device=device, dtype=torch.float32),
            torch.randn(100, device=device, dtype=torch.float32),
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
        lambda device: (torch.randn(100, device=device, dtype=torch.float32),),
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
        lambda device: (torch.randn(100, device=device, dtype=torch.bfloat16),),
        "v_add_f32 $0, $1, $1" if torch.version.hip else "add.f32 $0, $1, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x.float() * 2,
        compile_only=True,
        min_sm=80,
    ),
    AsmTestCase(
        "fp16_upcast",
        lambda device: (torch.randn(100, device=device, dtype=torch.float16),),
        "v_add_f32 $0, $1, $1" if torch.version.hip else "add.f32 $0, $1, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x.float() * 2,
        compile_only=True,
    ),
    # Integer operations
    AsmTestCase(
        "bitwise_and",
        lambda device: (
            torch.randint(0, 2**16, (100,), device=device, dtype=torch.int32),
            torch.randint(0, 2**16, (100,), device=device, dtype=torch.int32),
        ),
        "v_and_b32 $0, $1, $2" if torch.version.hip else "and.b32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=r,r,r",
        torch.int32,
        lambda x, y: x & y,
    ),
    AsmTestCase(
        "bitwise_or",
        lambda device: (
            torch.randint(0, 2**16, (100,), device=device, dtype=torch.int32),
            torch.randint(0, 2**16, (100,), device=device, dtype=torch.int32),
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
        lambda device: (
            torch.tensor([1.0, 2.0, 0.5, 16.0], device=device, dtype=torch.float32),
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
    # AMDGCN: use "v" and extract the lower 16 bits via v_bfe_u32.  With
    # real-true16, move the source's low half directly into the true16 output
    # with v_mov_b16 instead of using a 32-bit instruction.
    AsmTestCase(
        "truncate_to_uint16",
        lambda device: (
            torch.randint(0, 256, (100,), device=device, dtype=torch.int32),
        ),
        "v_bfe_u32 $0, $1, 0, 16" if torch.version.hip else "cvt.u16.u32 $0, $1;",
        "=v, v" if torch.version.hip else "=h,r",
        torch.uint16,
        lambda x: x.to(torch.uint16),
        compile_only=True,
        true16_asm_str="v_mov_b16 $0, $1.l",
    ),
    # Broadcasting
    AsmTestCase(
        "broadcast_add",
        lambda device: (
            torch.randn(4, 1, device=device, dtype=torch.float32),
            torch.randn(1, 8, device=device, dtype=torch.float32),
        ),
        "v_add_f32 $0, $1, $2" if torch.version.hip else "add.f32 $0, $1, $2;",
        "=v, v, v" if torch.version.hip else "=f,f,f",
        torch.float32,
        lambda x, y: x + y,
    ),
    # Non-contiguous
    AsmTestCase(
        "noncontiguous",
        lambda device: (torch.randn(8, 16, device=device, dtype=torch.float32).t(),),
        "v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
        "=v, v" if torch.version.hip else "=f,f",
        torch.float32,
        lambda x: x,
    ),
    # fp16/bf16 native asm (compile-only: inductor computes in fp32, needs downcast)
    # ROCm: Inductor feeds f32 values (upcasted for computation).  AMDGCN has no
    # "h" constraint for 16-bit regs, so we add in f32 and convert to the target
    # format.  PTX "h" constraints tell Triton to downcast before the asm.
    # Under real-true16 the fp16 output is allocated to a VGPR half, which
    # v_add_f32 cannot write; we route the f32 sum through physical v0 and
    # declare it as a clobber so LLVM does not reuse it across the asm block.
    AsmTestCase(
        "add_fp16_native",
        lambda device: (
            torch.randn(100, device=device, dtype=torch.float16),
            torch.randn(100, device=device, dtype=torch.float16),
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
        true16_asm_str="v_add_f32 v0, $1, $2\nv_cvt_f16_f32 $0, v0",
        true16_constraints="=v,v,v,~{v0}",
    ),
    # AMDGCN: v_cvt_pk_bf16_f32 packs two f32 values into bf16 in a single
    # 32-bit register.  We pass $0 twice — only the lower 16 bits (first
    # bf16 slot) are used by Triton.
    AsmTestCase(
        "add_bf16_native",
        lambda device: (
            torch.randn(100, device=device, dtype=torch.bfloat16),
            torch.randn(100, device=device, dtype=torch.bfloat16),
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
        lambda device: (torch.randn(128, device=device, dtype=torch.float32),),
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
        lambda device: (
            torch.randn(128, device=device, dtype=torch.float32),
            torch.randn(128, device=device, dtype=torch.float32),
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


def _require_device_triton(device, *, xfail=False):
    reason = None
    if not has_triton():
        reason = "Triton not available"
    else:
        try:
            device_interface = get_interface_for_device(torch.device(device).type)
        except NotImplementedError:
            reason = f"Triton not available for {device}"
        else:
            if not device_interface.is_triton_capable(device):
                reason = f"Triton not available for {device}"
            else:
                try:
                    device_interface.raise_if_triton_unavailable(device)
                except TritonUnavailableError as exc:
                    reason = str(exc)

    if reason is None:
        return
    if xfail:
        import pytest

        pytest.xfail(reason)
    raise unittest.SkipTest(reason)


@unittest.skipIf(not SM70OrLater, "Requires SM70+")
class TestInlineAsmElementwise(TestCase):
    """Parametrized tests for inline_asm_elementwise."""

    hw_classification = HardwareClassification.CUDA

    @parametrize(
        "case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i]
    )
    def test_eager_vs_compiled_bitwise(self, device, case_idx):
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

        inputs = tc.input_gen_fn(device)
        asm_str, constraints = _get_asm_config(tc)

        def fn(*args):
            return inline_asm_elementwise(
                *args,
                asm_str=asm_str,
                constraints=constraints,
                dtype=tc.dtype,
                pack=tc.pack,
            )

        # This test always runs torch.compile(inductor); Inductor needs Triton for CUDA.
        _require_device_triton(device)

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

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180228")
    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180131")
    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180124")
    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180116")
    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/180132")
    @parametrize(
        "case_idx", list(range(len(TEST_CASES))), name_fn=lambda i: TEST_CASE_NAMES[i]
    )
    def test_correctness(self, device, case_idx):
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

        inputs = tc.input_gen_fn(device)
        asm_str, constraints = _get_asm_config(tc)

        def fn(*args):
            return inline_asm_elementwise(
                *args,
                asm_str=asm_str,
                constraints=constraints,
                dtype=tc.dtype,
                pack=tc.pack,
            )

        if tc.compile_only:
            _require_device_triton(device)
            torch._dynamo.reset()
            result = torch.compile(fn, backend="inductor")(*inputs)
        else:
            result = fn(*inputs)
        expected = tc.approx_fn(*inputs)

        self.assertEqual(result.float(), expected.float(), atol=1e-5, rtol=1e-5)


class TestInlineAsmElementwiseInputValidation(TestCase):
    """Device-independent tests for error handling."""

    hw_classification = HardwareClassification.GENERIC

    def test_error_no_inputs(self):
        with self.assertRaises(ValueError):
            inline_asm_elementwise(
                asm_str="v_mov_b32 $0, 1.0"
                if torch.version.hip
                else "mov.f32 $0, 1.0;",
                constraints="=v" if torch.version.hip else "=f",
                dtype=torch.float32,
            )


class TestInlineAsmElementwiseErrors(TestCase):
    """CUDA-specific tests for error handling."""

    hw_classification = HardwareClassification.CUDA

    def test_error_cpu_tensor(self, device):
        x = torch.randn(100, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            inline_asm_elementwise(
                x,
                asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
                constraints="=v,v" if torch.version.hip else "=f,f",
                dtype=torch.float32,
            )

    def test_error_constraint_mismatch(self, device):
        x = torch.randn(100, device=device, dtype=torch.float32)
        y = torch.randn(100, device=device, dtype=torch.float32)
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

    def test_error_mixed_dtypes(self, device):
        x = torch.randn(100, device=device, dtype=torch.float32)
        y = torch.randint(0, 10, (100,), device=device, dtype=torch.int32)
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

    def test_error_multiple_outputs_require_compile(self, device):
        x = torch.arange(8, device=device, dtype=torch.int32)
        with self.assertRaisesRegex(
            RuntimeError, "requires torch.compile.*multiple outputs"
        ):
            inline_asm_elementwise(
                x,
                asm_str="mov.b32 $0, $2; mov.b32 $1, $2;",
                constraints="=r,=r,r",
                dtype=(torch.int32, torch.int32),
            )

    def test_error_multiple_output_constraint_mismatch(self, device):
        x = torch.arange(8, device=device, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "Expected 2 output constraint"):
            inline_asm_elementwise(
                x,
                asm_str="mov.b32 $0, $1;",
                constraints="=r,r",
                dtype=(torch.int32, torch.int32),
            )


class TestInlineAsmElementwiseMultipleOutputs(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _check_stochastic_rounding(self, device, fn, rng_state):
        halfway = torch.tensor([0x3F808000], device=device, dtype=torch.int32)
        x = halfway.expand(2).view(torch.float32)

        from torch._inductor.utils import run_and_get_code

        result, sources = run_and_get_code(
            torch.compile(fn, fullgraph=True), x, rng_state
        )
        self.assertEqual(
            result,
            torch.tensor([1.0078125, 1.0], device=device, dtype=torch.bfloat16),
        )
        code = "\n".join(sources)
        self.assertEqual(code.count("@triton_heuristics"), 1)
        self.assertEqual(code.count("tl.inline_asm_elementwise("), 1)

    @xfailIfNoAcceleratorTriton
    @skipIfRocm(msg="PTX test")
    def test_multiple_outputs_compile(self, device):
        _require_device_triton(device, xfail=True)

        def asm(x):
            return inline_asm_elementwise(
                x,
                asm_str="mov.b32 $0, $2; cvt.rn.f32.s32 $1, $2;",
                constraints="=r,=f,r",
                dtype=(torch.int32, torch.float32),
            )

        def fn(x):
            first, second = asm(x)
            return first.float() * 3 + second

        x = torch.arange(128, device=device, dtype=torch.int32)
        from torch._inductor.utils import run_and_get_code

        outputs, output_sources = run_and_get_code(
            torch.compile(asm, fullgraph=True), x
        )
        self.assertEqual(outputs, (x, x.float()))
        output_code = "\n".join(output_sources)
        self.assertEqual(output_code.count("@triton_heuristics"), 1)
        self.assertEqual(output_code.count("tl.inline_asm_elementwise("), 1)

        result, sources = run_and_get_code(torch.compile(fn, fullgraph=True), x)

        self.assertEqual(result, x.float() * 4)
        self.assertEqual("\n".join(sources).count("tl.inline_asm_elementwise("), 1)

    @xfailIfNoAcceleratorTriton
    @skipIfRocm(msg="PTX test")
    def test_stochastic_rounding_with_rng_input(self, device):
        _require_device_triton(device, xfail=True)

        def fn(x, rng):
            lower, remainder = inline_asm_elementwise(
                x,
                asm_str=(
                    "mov.b32 $0, $2; and.b32 $0, $0, 0xffff0000; "
                    "mov.b32 $1, $2; and.b32 $1, $1, 0xffff;"
                ),
                constraints="=r,=r,f",
                dtype=(torch.int32, torch.int32),
            )
            round_up = (rng & 0xFFFF) < remainder
            rounded = lower + round_up.to(torch.int32) * 0x10000
            return rounded.view(torch.float32).to(torch.bfloat16)

        rng = torch.tensor([0x7FFF, 0x8000], device=device, dtype=torch.int32)
        self._check_stochastic_rounding(device, fn, rng)

    @xfailIfNoAcceleratorTriton
    @skipIfRocm(msg="PTX test")
    def test_stochastic_rounding_with_inline_rng(self, device):
        _require_device_triton(device, xfail=True)

        def fn(x, counter):
            lower, remainder, rng = inline_asm_elementwise(
                x,
                counter,
                asm_str=(
                    "mov.b32 $0, $3; and.b32 $0, $0, 0xffff0000; "
                    "mov.b32 $1, $3; and.b32 $1, $1, 0xffff; "
                    "mad.lo.u32 $2, $4, 1664525, 1013904223;"
                ),
                constraints="=r,=r,=r,f,r",
                dtype=(torch.int32, torch.int32, torch.int32),
            )
            round_up = (rng & 0xFFFF) < remainder
            rounded = lower + round_up.to(torch.int32) * 0x10000
            return rounded.view(torch.float32).to(torch.bfloat16)

        counter = torch.tensor([1, 0], device=device, dtype=torch.int32)
        self._check_stochastic_rounding(device, fn, counter)

    @xfailIfNoAcceleratorTriton
    @skipIfRocm(msg="PTX test")
    def test_multiple_outputs_with_pack_compile(self, device):
        _require_device_triton(device, xfail=True)

        def fn(x):
            return inline_asm_elementwise(
                x,
                asm_str=(
                    "mov.b32 $0, $4; mov.b32 $1, $5; "
                    "add.s32 $2, $4, 1; add.s32 $3, $5, 1;"
                ),
                constraints="=r,=r,=r,=r,r,r",
                dtype=(torch.int32, torch.int32),
                pack=2,
            )

        x = torch.arange(128, device=device, dtype=torch.int32)
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), (x, x + 1))


@unittest.skipIf(not SM70OrLater, "Requires SM70+")
class TestInlineAsmElementwiseEdgeCases(TestCase):
    """Tests for edge cases."""

    hw_classification = HardwareClassification.CUDA

    def test_empty_tensor(self, device):
        x = torch.empty(0, device=device, dtype=torch.float32)
        result = inline_asm_elementwise(
            x,
            asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
            constraints="=v, v" if torch.version.hip else "=f,f",
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, torch.Size([0]))

    def test_scalar_tensor(self, device):
        x = torch.tensor(3.14, device=device, dtype=torch.float32)
        result = inline_asm_elementwise(
            x,
            asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
            constraints="=v, v" if torch.version.hip else "=f,f",
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, torch.Size([]))
        self.assertEqual(result, x)

    def test_4d_tensor(self, device):
        x = torch.randn(2, 3, 4, 5, device=device, dtype=torch.float32)
        result = inline_asm_elementwise(
            x,
            asm_str="v_mov_b32 $0, $1" if torch.version.hip else "mov.f32 $0, $1;",
            constraints="=v, v" if torch.version.hip else "=f,f",
            dtype=torch.float32,
        )
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result, x)

    @xfailIfNoAcceleratorTriton
    def test_composition_with_pytorch_ops(self, device):
        _require_device_triton(device, xfail=True)

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

        x = torch.randn(100, device=device, dtype=torch.float32)
        y = torch.randn(100, device=device, dtype=torch.float32)

        eager_result = fn(x, y)
        compiled_fn = torch.compile(fn, backend="inductor")
        compiled_result = compiled_fn(x, y)

        self.assertEqual(eager_result, compiled_result)
        self.assertEqual(eager_result, x * 2 + y + 1.0)

    def test_output_strides_mixed_inputs(self, device):
        """Verify fake mode output strides match eager (TensorIterator) strides."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Two inputs with different strides: one contiguous, one transposed.
        # This exercises TensorIterator's slow path for stride computation.
        x = torch.randn(8, 16, device=device, dtype=torch.float32)
        y = torch.randn(16, 8, device=device, dtype=torch.float32).t()

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

    def test_vmap(self, device):
        asm_str = "v_add_f32 $0, $1, $2" if torch.version.hip else "add.f32 $0, $1, $2;"
        constraints = "=v, v, v" if torch.version.hip else "=f,f,f"

        def add(a, b):
            return inline_asm_elementwise(
                a, b, asm_str=asm_str, constraints=constraints, dtype=torch.float32
            )

        bias = torch.randn(64, device=device)
        x = torch.randn(8, 64, device=device)
        # batched vector + unbatched vector
        self.assertEqual(torch.vmap(add, in_dims=(0, None))(x, bias), x + bias)
        # batched scalar + unbatched vector: batch dim must not be consumed by
        # trailing-dim broadcasting
        y = torch.randn(8, device=device)
        self.assertEqual(torch.vmap(add, in_dims=(0, None))(y, bias), y[:, None] + bias)
        # non-zero batch dim
        self.assertEqual(
            torch.vmap(add, in_dims=(1, None))(x, bias[:8]), x.t() + bias[:8]
        )
        # nested vmap down to scalars
        nested = torch.vmap(torch.vmap(add))(x, x)
        self.assertEqual(nested, x + x)

    @xfailIfNoAcceleratorTriton
    def test_dynamic_shapes(self, device):
        _require_device_triton(device, xfail=True)

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
            x = torch.randn(size, device=device, dtype=torch.float32)
            y = torch.randn(size, device=device, dtype=torch.float32)
            eager_result = fn(x, y)
            compiled_result = compiled_fn(x, y)
            self.assertEqual(eager_result, compiled_result)


@unittest.skipIf(not SM70OrLater, "Requires SM70+")
@xfailIfNoAcceleratorTriton
class TestInlineAsmPackPadding(TestCase):
    """Test that pack padding works when block size < pack."""

    hw_classification = HardwareClassification.CUDA

    def test_pack2_xblock1_padding(self, device):
        """Force XBLOCK=1 with pack=2 so padding is needed."""
        _require_device_triton(device, xfail=True)

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

        x = torch.randn(128, device=device, dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertEqual(result, x)
        # Verify padding helpers are emitted in the generated code
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)

    def test_pack4_xblock1_padding(self, device):
        """Force XBLOCK=1 with pack=4 so padding is needed."""
        _require_device_triton(device, xfail=True)

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

        x = torch.randn(128, device=device, dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertEqual(result, x)
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)

    def test_pack4_xblock2_partial_padding(self, device):
        """XBLOCK=2 < pack=4, so 1 round of padding is needed (not 2)."""
        _require_device_triton(device, xfail=True)

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

        x = torch.randn(128, device=device, dtype=torch.float32)
        with torch._inductor.virtualized.V.set_choices_handler(ForceXBlock2()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(torch.compile(fn, backend="inductor"), x)

        self.assertEqual(result, x)
        FileCheck().check("inline_asm_pack").check("inline_asm_unpack").run(code)

    def test_pack2_xblock1_yblock1_padding(self, device):
        """Force XBLOCK=1, YBLOCK=1 with pack=2 on a 2D-tiled kernel."""
        _require_device_triton(device, xfail=True)

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

        x = torch.randn(8, 16, device=device, dtype=torch.float32)
        # Transposed input triggers 2D tiling (different stride patterns)
        y = torch.randn(16, 8, device=device, dtype=torch.float32).T
        with torch._inductor.virtualized.V.set_choices_handler(ForceXY1()):
            torch._dynamo.reset()
            result, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor"), x, y
            )

        self.assertEqual(result, x + y)
        FileCheck().check("YBLOCK").check("inline_asm_pack").check(
            "inline_asm_unpack"
        ).run(code)


instantiate_device_type_tests(TestInlineAsmElementwise, globals(), only_for=("cuda",))
instantiate_device_type_tests(
    TestInlineAsmElementwiseErrors, globals(), only_for=("cuda",)
)
instantiate_device_type_tests(
    TestInlineAsmElementwiseEdgeCases, globals(), only_for=("cuda",)
)
instantiate_device_type_tests(TestInlineAsmPackPadding, globals(), only_for=("cuda",))
instantiate_device_type_tests(
    TestInlineAsmElementwiseMultipleOutputs, globals(), only_for=("cuda",)
)


if __name__ == "__main__":
    run_tests()
