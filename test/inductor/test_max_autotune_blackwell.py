# Owner(s): ["module: inductor"]
import ast
import contextlib
import dataclasses
import re
import unittest
from unittest import mock

import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.autows_utils import has_two_ctas, meta_ws_enabled
from torch._inductor.codegen import simd
from torch._inductor.heuristics.registry import (
    _HEURISTIC_CACHE,
    get_template_heuristic,
    override_template_heuristics,
)
from torch._inductor.heuristics.template.bmm import (
    CUDABlackwellBMMTemplateConfigHeuristic,
)
from torch._inductor.heuristics.template.triton import (
    _use_template_autows,
    BaseHeuristicSingleton,
    BlackwellGPUGemmConfig,
    BlackwellTMATemplateConfigMixin,
    CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic,
    CUDABlackwellPersistentTMATemplateConfigHeuristic,
    CUDAScaledBlackwellTMATemplateConfigHeuristic,
    GemmConfig,
    TMATemplateConfigMixin,
)
from torch._inductor.ir import FixedLayout, MultiTemplateBuffer
from torch._inductor.kernel.bmm import (
    blackwell_ws_persistent_tma_bmm_template,
    BlackwellBMMConfig,
)
from torch._inductor.kernel.mm import (
    blackwell_ws_persistent_tma_mm_template,
    persistent_tma_mm_template,
    scaled_mm_device_tma_epilogue_scaling_template,
    scaled_mm_device_tma_main_loop_scaling_template,
)
from torch._inductor.kernel.mm_common import blackwell_persistent_mm_grid
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.lowering import lowerings
from torch._inductor.scheduler import Scheduler
from torch._inductor.select_algorithm import NoValidChoicesError
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import get_num_sms, run_and_get_code
from torch._inductor.virtualized import V
from torch.nn.functional import scaled_mm, ScalingType  # type: ignore[attr-defined]
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._triton import (
    has_datacenter_blackwell_tma_device,
    has_triton_tma_device,
)


def has_tlx() -> bool:
    """Check if TLX (Triton Language eXtensions) is available."""
    try:
        import triton.language.extra.tlx  # noqa: F401

        return True
    except ImportError:
        return False


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


@torch.library.custom_op("inductor_test::blackwell_bmm", mutates_args={})
def blackwell_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b)


@blackwell_bmm.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], a.shape[1], b.shape[2]))


@torch.library.custom_op("inductor_test::blackwell_bmm_flat", mutates_args={})
def blackwell_bmm_flat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b).flatten(0, 1)


@blackwell_bmm_flat.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0] * a.shape[1], b.shape[2]))


class _Test2CTABlackwellBMMHeuristic(CUDABlackwellBMMTemplateConfigHeuristic):
    bmm_configs = (BlackwellBMMConfig(128, 128, 64, 4, 8, two_ctas=True),)


class _Test1CTABlackwellBMMHeuristic(CUDABlackwellBMMTemplateConfigHeuristic):
    bmm_configs = (BlackwellBMMConfig(128, 128, 64, 4, 8),)


def setUpModule():
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")


def tearDownModule():
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


@instantiate_parametrized_tests
class TestMaxAutotuneBlackwell(TestCase):
    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("epilogue_subtile", (1, 2, 4))
    @parametrize("host_side_tma", (False, True))
    def test_blackwell_max_autotune_regular_mm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
        epilogue_subtile: int,
        host_side_tma: bool,
    ):
        def mm(a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)
            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        M, N, K = 32, 16, 48
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )

        epilogue_subtile_regex = f"EPILOGUE_SUBTILE={epilogue_subtile}"
        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "triton.enable_host_side_tma": host_side_tma,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                "test_configs.autotune_choice_desc_regex": epilogue_subtile_regex,
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        fc = FileCheck().check("triton_tem_fused_mm")
        # Dynamic dims fall back to device-side descriptors.
        if host_side_tma and not dynamic:
            fc.check("host_tma_descriptor_args")
            if not tma_store:
                fc.check_not("tl.make_tensor_descriptor")
        else:
            fc.check("tl.make_tensor_descriptor")
        fc.check(write_api).run(code[0])
        if host_side_tma and tma_store and not dynamic:
            # Loads are host-side, so the TMA store is the only descriptor
            # still built in-kernel.
            FileCheck().check_count("tl.make_tensor_descriptor", 1, exactly=True).run(
                code[0]
            )
            FileCheck().check("tl.make_tensor_descriptor(out_ptr0").run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @largeTensorTest("20 GB", device=GPU_TYPE)
    def test_blackwell_mm_tma_store_large_output_int64_indexing(self):
        """An output past 2^31 elements makes the kernel index in int64; the
        Blackwell template must still be offered with its TMA store."""
        M, N, K = 5592406, 384, 128
        self.assertGreater(M * N, 2**31 - 1)
        # Entries in {-1, 0, 1} keep every partial sum exact, so any config must match bitwise.
        a = torch.randint(-1, 2, (M, K), device=GPU_TYPE, dtype=torch.int8).bfloat16()
        b = torch.randint(-1, 2, (K, N), device=GPU_TYPE, dtype=torch.int8).bfloat16()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": True,
                "triton.num_decompose_k_splits": 0,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            actual, code = run_and_get_code(torch.compile(torch.mm), a, b)

        FileCheck().check("INDEX_DTYPE : tl.constexpr = tl.int64").check(
            "tma_descriptor0.store("
        ).run(code[0])
        self.assertEqual(actual, torch.mm(a, b), atol=0, rtol=0)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("host_side_tma", (False, True))
    def test_blackwell_square_operand_dim_order(
        self, a_transposed: bool, b_transposed: bool, host_side_tma: bool
    ):
        # A square operand has the same descriptor shape under either dim_order,
        # so only the strides distinguish row-major from transposed. Picking the
        # wrong order is invisible in the shape and shows up as a wrong result.
        S = 512
        a = torch.randn(S, S).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(S, S).to(torch.float16).to(GPU_TYPE)

        def mm(a, b):
            return torch.mm(a.T if a_transposed else a, b.T if b_transposed else b)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_host_side_tma": host_side_tma,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            actual, code = run_and_get_code(torch.compile(mm), a, b)
        torch.testing.assert_close(actual, mm(a, b), atol=1e-2, rtol=1e-2)
        fc = FileCheck().check("triton_tem_fused")
        if host_side_tma:
            fc.check("host_tma_descriptor_args").check_not("tl.make_tensor_descriptor")
        else:
            fc.check("tl.make_tensor_descriptor")
        fc.run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", ("mm", "addmm"))
    def test_blackwell_host_side_tma_transposed_b(self, op: str):
        # Regression test for host-side TMA with a column-major / transposed B
        # operand (B_ROW_MAJOR=False, the nn.Linear `x @ W.t()` case).
        # Previously the host launcher re-permuted the runtime tensor's own
        # dims -- which are in base layout for a transposed operand -- producing
        # an incorrect result. Also covers addmm host-side TMA, previously
        # untested.
        M, N, K = 512, 256, 512
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        # b is [N, K]; b.t() is the [K, N] column-major operand.
        b = torch.randn(N, K).to(torch.float16).to(GPU_TYPE)
        bias = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        def fn(a, b, bias):
            if op == "addmm":
                return torch.addmm(bias, a, b.t())
            return torch.mm(a, b.t())

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_host_side_tma": True,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(fn), a, b, bias)
        c_expected = fn(a, b, bias)
        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)
        # host-side TMA: descriptors come from the launcher, none built in-kernel
        FileCheck().check("triton_tem_fused").check_not(
            "tl.make_tensor_descriptor"
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("host_side_tma", (False, True))
    def test_blackwell_size_one_dim_persistent_tma(self, host_side_tma: bool):
        # A contiguous [1, K] operand is reported transposed by
        # Layout.is_transposed(), which skips size-1 dims, so the descriptor was
        # built with a non-unit trailing stride.
        M, N, K = 1, 512, 1024
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_host_side_tma": host_side_tma,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(torch.mm), a, b)
        torch.testing.assert_close(c_actual, torch.mm(a, b), atol=1e-2, rtol=1e-2)
        fc = FileCheck().check("triton_tem_fused_mm")
        if host_side_tma:
            fc.check("host_tma_descriptor_args")
            fc.check_not("tl.make_tensor_descriptor(base=")
        else:
            fc.check("tl.make_tensor_descriptor(base=")
        fc.run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_host_side_tma_one_dynamic_operand(self):
        # Only A has a dynamic dim: the launcher reads it at launch, so both
        # operands keep host-side descriptors and none is built in-kernel.
        M, N, K = 1024, 256, 512
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_host_side_tma": True,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(torch.mm), a, b)
        torch.testing.assert_close(c_actual, torch.mm(a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_tem_fused_mm").check_not(
            "tl.make_tensor_descriptor(base="
        ).run(code[0])
        FileCheck().check("'host_tma_descriptor_args'").check("'arg_A'").check(
            "'launch_args': ['ks0']"
        ).run(code[0])

    # NOTE: the current Inductor template verifies that the scaling mode is either per-tensor or per-row
    # TODO: support additional scaling modes for Blackwell

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_host_side_tma_aliased_dynamic_operand(self):
        # Both operands view x, so def_kernel passes them as one arg. A is
        # dynamic, so B must also take a device-side descriptor: the launcher
        # can't pass the shared arg as a host TensorDescriptor to only one.
        x = torch.randint(-2, 3, (256, 64)).to(torch.float16).to(GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)

        def mm(x):
            return x @ x[:64].t()

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_host_side_tma": True,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm), x)
        torch.testing.assert_close(c_actual, mm(x), atol=0, rtol=0)
        FileCheck().check_not("host_tma_descriptor_args").run(code[0])
        FileCheck().check("tl.make_tensor_descriptor(base=A").check(
            "tl.make_tensor_descriptor(base=B"
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("host_side_tma", (False, True))
    def test_blackwell_max_autotune_scaled_mm_per_tensor_persistent_tma(
        self,
        dynamic: bool,
        tma_store: bool,
        host_side_tma: bool,
    ):
        def scaled_mm(a, b, scale_a, scale_b):
            # NOTE: Inductor constrains a to be row_major and b to be col_major
            return torch._scaled_mm(
                a, b.t(), scale_a, scale_b, use_fast_accum=True, out_dtype=torch.float16
            )

        def get_scale_per_tensor(t):
            scale = torch.finfo(torch.float8_e4m3fn).max / t.abs().max()
            return scale.to(torch.float32)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte.
        M, N, K = 32, 16, 48
        a = (torch.randn((M, K)).to(torch.float16).to(GPU_TYPE)).repeat(8, 8)
        b = (torch.randn((N, K)).to(torch.float16).to(GPU_TYPE)).repeat(8, 8)

        scale_a = get_scale_per_tensor(a)
        scale_b = get_scale_per_tensor(b)

        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "triton.enable_host_side_tma": host_side_tma,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(scaled_mm, dynamic=dynamic), a, b, scale_a, scale_b
            )
            c_expected = scaled_mm(a, b, scale_a, scale_b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=0.5)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        fc = FileCheck().check("triton_tem_fused__scaled_mm")
        # Dynamic dims fall back to device-side descriptors.
        if host_side_tma and not dynamic:
            fc.check("host_tma_descriptor_args")
        else:
            fc.check("tl.make_tensor_descriptor(base=")
        fc.check("_desc.load(").check(write_api).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("host_side_tma", (False, True))
    def test_blackwell_max_autotune_scaled_mm_per_row_persistent_tma(
        self,
        dynamic: bool,
        tma_store: bool,
        host_side_tma: bool,
    ):
        def scaled_mm(a, b, scale_a, scale_b):
            # NOTE: Inductor constrains a to be row_major and b to be col_majo
            return torch._scaled_mm(
                a,
                b.t(),
                scale_a,
                scale_b.t(),
                use_fast_accum=True,
                out_dtype=torch.bfloat16,
            )

        def get_scale_per_row(t):
            scale = (
                torch.finfo(torch.float8_e4m3fn).max
                / t.abs().max(dim=1, keepdim=True).values
            )
            return scale.to(torch.float32)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte.
        M, N, K = 32, 16, 48
        a = (torch.randn((M, K)).to(torch.bfloat16).to(GPU_TYPE)).repeat(8, 8)
        b = (torch.randn((N, K)).to(torch.bfloat16).to(GPU_TYPE)).repeat(8, 8)

        scale_a = get_scale_per_row(a)
        scale_b = get_scale_per_row(b)

        a = a.to(torch.float8_e4m3fn)
        b = b.to(torch.float8_e4m3fn)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "triton.enable_host_side_tma": host_side_tma,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(scaled_mm, dynamic=dynamic), a, b, scale_a, scale_b
            )
            c_expected = scaled_mm(a, b, scale_a, scale_b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=0.5)
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"
        fc = FileCheck().check("triton_tem_fused__scaled_mm")
        # Dynamic dims fall back to device-side descriptors.
        if host_side_tma and not dynamic:
            fc.check("host_tma_descriptor_args")
        else:
            fc.check("tl.make_tensor_descriptor(base=")
        fc.check("_desc.load(").check(write_api).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    @parametrize("epilogue_subtile", (1, 2, 4))
    @parametrize("host_side_tma", (False, True))
    def test_blackwell_max_autotune_addmm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
        epilogue_subtile: int,
        host_side_tma: bool,
    ):
        def addmm(x, a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            x = x.repeat(8)
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        epilogue_subtile_regex = f"EPILOGUE_SUBTILE={epilogue_subtile}"
        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": tma_store,
                "triton.enable_host_side_tma": host_side_tma,
                "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                "test_configs.autotune_choice_desc_regex": epilogue_subtile_regex,
                # If we dynamically disable pipelining,
                # triton_blackwell_ws_persistent_tma template will
                # be picked and then cause mis-aligned memory access.
                # If we don't dynamically disable pipelining,
                # this template is skipped and triton_tem_fused_addmm_repeat_3
                # is used.
                #
                # Fundamentally we should fix the triton_blackwell_ws_persistent_tma template
                # The flag below work around the problem.
                #
                # This only happens in fbcode https://www.internalfb.com/diff/D101855575.
                # Can not repro it in OSS build.
                "triton.dynamic_disable_pipelining": not config.is_fbcode(),
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(addmm, dynamic=dynamic), x, a, b
            )
            c_expected = addmm(x, a, b)

        make_desc_api = "tl.make_tensor_descriptor"
        read_api = "_desc.load("
        if tma_store:
            # Verify that we are using a TMA implementation
            # Note: The tma_descriptor0 is generated by the kernel. If the
            # code generation process changes this could change.
            write_api = "tma_descriptor0.store"
        else:
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        fc = FileCheck().check("triton_tem_fused_addmm")
        # Dynamic dims fall back to device-side descriptors.
        if host_side_tma and not dynamic:
            fc.check("host_tma_descriptor_args")
            if not tma_store:
                fc.check_not(make_desc_api)
        else:
            fc.check(make_desc_api)
        fc.check(read_api).check(write_api).run(code[0])
        if host_side_tma and tma_store and not dynamic:
            # Loads are host-side, so the TMA store is the only descriptor
            # still built in-kernel.
            FileCheck().check_count(make_desc_api, 1, exactly=True).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    def test_resolved_host_tma_descriptor_args_symbolic_block_shape(self):
        from types import SimpleNamespace

        from torch._inductor.codegen.triton import TritonKernel

        # A block shape naming an autotuned kernel arg stays a name for the
        # launcher to look up per config; anything else resolves to a value.
        signature = [SimpleNamespace(name="XBLOCK")]
        kernel = SimpleNamespace(
            args=SimpleNamespace(python_argdefs=lambda: ([], [], signature, [])),
            persistent_reduction=False,
            host_tma_descriptor_args={
                "in_ptr0": SimpleNamespace(
                    block_shape=["XBLOCK", 128, "YBLOCK"],
                    shape=[1024, "s0"],
                    strides=["s0", 1],
                )
            },
        )

        resolved = TritonKernel.resolved_host_tma_descriptor_args(kernel)

        self.assertEqual(
            resolved["in_ptr0"],
            {
                "block_shape": ["XBLOCK", 128, "YBLOCK"],
                "shape": [1024, "s0"],
                "strides": ["s0", 1],
            },
        )
        # An already-resolved dict passes through untouched.
        kernel.host_tma_descriptor_args = {"in_ptr1": {"block_shape": [64]}}
        self.assertEqual(
            TritonKernel.resolved_host_tma_descriptor_args(kernel),
            {"in_ptr1": {"block_shape": [64]}},
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_host_side_tma_signature_upgraded_at_precompile(self):
        from torch._inductor.runtime import triton_heuristics

        def mm(a, b):
            return torch.mm(a, b)

        a = torch.randn(1024, 1024).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(1024, 1024).to(torch.float16).to(GPU_TYPE)

        captured: list[dict] = []
        orig = triton_heuristics.CachingAutotuner._create_compile_meta

        def _spy(self, cfg):
            meta = orig(self, cfg)
            if self.inductor_meta.get("host_tma_descriptor_args"):
                captured.append(dict(meta["signature"]))
            return meta

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_host_side_tma": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                }
            ),
            mock.patch.object(
                triton_heuristics.CachingAutotuner, "_create_compile_meta", _spy
            ),
        ):
            actual, code = run_and_get_code(torch.compile(mm), a, b)

        torch.testing.assert_close(actual, mm(a, b), atol=1e-2, rtol=1e-2)
        self.assertTrue(captured, "no host-side TMA kernel was precompiled")
        upgraded = [
            ty
            for sig in captured
            for ty in sig.values()
            if isinstance(ty, str) and ty.startswith("tensordesc<")
        ]
        self.assertTrue(upgraded, f"no arg upgraded to tensordesc<>: {captured}")
        # Block dims must be concrete by this point: the launcher resolves the
        # autotuned symbol per config.
        FileCheck().check_regex(r"tensordesc<\w+\[\d+, \d+\]>").run("\n".join(upgraded))
        # The upgrade is the launcher's job now, so codegen must not have
        # already baked it into the generated module.
        FileCheck().check_not("tensordesc<").run(code[0])


@instantiate_parametrized_tests
@unittest.skipIf(
    config.triton.enable_host_side_tma,
    "epilogue fusion registers a descriptor the host path cannot resolve",
)
class TestBlackwellTMAStoreFusion(TestCase):
    """Tests for TMA store with fused pointwise epilogues on Blackwell."""

    @staticmethod
    def _make_tma_store_test_config(epilogue_subtile: int) -> BlackwellGPUGemmConfig:
        """FLATTEN=False test config for TMA store pointwise epilogue fusion.

        The WS pipeline doesn't yet support FLATTEN=True + TMA descriptor store.
        """
        return BlackwellGPUGemmConfig(
            128,
            128,
            64,
            3,
            4,
            epilogue_subtile=epilogue_subtile,
            warp_specialize=False,
            flatten=False,
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_mm_sigmoid_epilogue_fusion_tma_store(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        """Verify fused mm + sigmoid epilogue with TMA store (FLATTEN=False)."""

        def fn(x, W):
            mm = torch.mm(x, W.T)
            return x * (2.0 * torch.sigmoid(mm))

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_store_test_config(epilogue_subtile)
        # Ensure the heuristic cache is populated, then patch mm_configs.
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_template_tma_store": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W)
                expected = fn(x, W)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused_mm").check("tma_descriptor").check(
            ".store"
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_addmm_relu_epilogue_fusion_tma_store(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        """Verify fused addmm + relu epilogue with TMA store (FLATTEN=False)."""

        def fn(x, W, bias):
            mm = torch.mm(x, W.T)
            return torch.relu(mm + bias)

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)
        bias = torch.randn(N, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_store_test_config(epilogue_subtile)
        # Ensure the heuristic cache is populated, then patch mm_configs.
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_template_tma_store": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W, bias)
                expected = fn(x, W, bias)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused").check("tma_descriptor").check(
            ".store"
        ).run(code[0])


@instantiate_parametrized_tests
@unittest.skipIf(
    config.triton.enable_host_side_tma,
    "epilogue fusion registers a descriptor the host path cannot resolve",
)
class TestBlackwellTMALoadFusion(TestCase):
    """Tests for TMA load with fused pointwise epilogues on Blackwell."""

    @staticmethod
    def _make_tma_load_test_config(epilogue_subtile: int) -> BlackwellGPUGemmConfig:
        return BlackwellGPUGemmConfig(
            128,
            128,
            64,
            3,
            4,
            epilogue_subtile=epilogue_subtile,
            warp_specialize=False,
            flatten=False,
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_mm_sigmoid_epilogue_tma_load(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        """Verify fused mm + sigmoid epilogue with TMA load (1 epilogue load)."""

        def fn(x, W):
            mm = torch.mm(x, W.T)
            return x * (2.0 * torch.sigmoid(mm))

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_load_test_config(epilogue_subtile=epilogue_subtile)
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_tma_load_for_template_epilogue": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W)
                expected = fn(x, W)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused_mm").check("tma_descriptor").check(
            ".load(["
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_mm_scale_bias_epilogue_tma_load(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        """Verify fused mm * scale + bias epilogue with TMA load (2 epilogue loads)."""

        def fn(x, W, scale, bias):
            return torch.mm(x, W.T) * scale + bias

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)
        scale = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE)
        bias = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_load_test_config(epilogue_subtile=epilogue_subtile)
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_tma_load_for_template_epilogue": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), x, W, scale, bias)
                expected = fn(x, W, scale, bias)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused").check("tma_descriptor").check(
            ".load(["
        ).run(code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("shape", ((512, 256, 256), (4096, 256, 256)))
    @parametrize("epilogue_subtile", (1, 2))
    def test_blackwell_addmm_1d_bias_epilogue_tma_load(
        self,
        shape: tuple[int, int, int],
        epilogue_subtile: int,
    ):
        """Verify fused addmm with 1D bias epilogue with TMA load."""

        def fn(bias, x, W):
            return torch.addmm(bias, x, W.T)

        M, N, K = shape
        x = torch.randn(M, K, dtype=torch.bfloat16, device=GPU_TYPE)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=GPU_TYPE)
        bias = torch.randn(N, dtype=torch.bfloat16, device=GPU_TYPE)

        test_config = self._make_tma_load_test_config(epilogue_subtile=epilogue_subtile)
        from torch._inductor.heuristics.template.registry import get_template_heuristic

        _cache_keys = [
            ("triton::blackwell_ws_persistent_tma", "cuda", "mm"),
            ("triton::blackwell_ws_persistent_tma", "cuda", "addmm"),
        ]
        orig_configs_by_key = {}
        for key in _cache_keys:
            h = get_template_heuristic(*key)
            orig_configs_by_key[key] = h.mm_configs
            h.mm_configs = [test_config]
        try:
            with config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.enable_tma_load_for_template_epilogue": True,
                    "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                }
            ):
                actual, code = run_and_get_code(torch.compile(fn), bias, x, W)
                expected = fn(bias, x, W)
        finally:
            for key, orig in orig_configs_by_key.items():
                if key in _HEURISTIC_CACHE:
                    _HEURISTIC_CACHE[key].mm_configs = orig

        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)

        FileCheck().check("triton_tem_fused").check("tma_descriptor").check(
            ".load(["
        ).run(code[0])

    @staticmethod
    def _run_with_mm_config(
        fn,
        args,
        test_config,
        *extra_configs,
        autows=(1, False),
        dynamic=None,
        fused_ms=0.0,
        unfused_ms=1e-9,
        **patches,
    ):
        """Compile fn with only the given configs as Triton choices. Under
        template autoWS, only the (DATA_PARTITION_FACTOR, TWO_CTAS) = autows
        variants of them. Fused epilogue benchmarks report fused_ms and unfused
        node benchmarks unfused_ms (None times them for real). Both are tiny by
        default, so every fusion the gates allow is kept regardless of GPU timing
        noise; unfused_ms stays positive so equal timings don't reject a fusion."""
        keys = [
            ("triton::blackwell_ws_persistent_tma", "cuda", op)
            for op in ("mm", "addmm")
        ]
        heuristics = [get_template_heuristic(*key) for key in keys]
        # Template autoWS builds its configs from blackwell_persistent_mm_configs.
        orig = [(h.mm_configs, h.blackwell_persistent_mm_configs) for h in heuristics]
        configs = [
            test_config,
            # Epilogues are only benchmarked, and so reductions only fused,
            # when autotuning has more than one choice. This twin hosts the
            # same reductions.
            dataclasses.replace(test_config, num_stages=test_config.num_stages - 1),
            *extra_configs,
        ]
        for heuristic in heuristics:
            heuristic.mm_configs = heuristic.blackwell_persistent_mm_configs = configs

        # mm and addmm override _generate_autows_configs with their own tuned
        # sets, so build the autoWS variants of the pinned configs by the base rule.
        base_autows_configs = BlackwellTMATemplateConfigMixin._generate_autows_configs

        def autows_configs(self):
            return [
                cfg
                for cfg in base_autows_configs(self)
                if (cfg.data_partition_factor, cfg.two_ctas) == autows
            ]

        pin_autows = contextlib.ExitStack()
        for cls in {type(heuristic) for heuristic in heuristics}:
            pin_autows.enter_context(
                mock.patch.object(cls, "_generate_autows_configs", autows_configs)
            )

        try:
            with (
                config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "TRITON",
                        "triton.enable_persistent_tma_matmul": True,
                        "test_configs.autotune_choice_name_regex": "blackwell_ws_persistent_tma",
                        "benchmark_template_fusion": True,
                        **patches,
                    }
                ),
                pin_autows,
                mock.patch.object(
                    Scheduler, "benchmark_codegened_module", return_value=(fused_ms, "")
                ),
                (
                    contextlib.nullcontext()
                    if unfused_ms is None
                    else mock.patch.object(
                        Scheduler,
                        "benchmark_fused_nodes",
                        return_value=(unfused_ms, ""),
                    )
                ),
            ):
                return run_and_get_code(torch.compile(fn, dynamic=dynamic), *args)
        finally:
            for key, configs in zip(keys, orig):
                if (cached := _HEURISTIC_CACHE.get(key)) is not None:
                    cached.mm_configs, cached.blackwell_persistent_mm_configs = configs

    @staticmethod
    def _poison_outputs():
        """Poison allocations so unwritten outputs show up as NaN."""

        def empty_strided(size, stride, dtype):
            out = torch.empty_strided(size, stride, dtype=dtype, device=GPU_TYPE)
            out.untyped_storage().fill_(255)
            return out

        return mock.patch("torch._C._dynamo.guards._empty_strided_cuda", empty_strided)

    @staticmethod
    def _with_kernel_harnesses(code: str) -> str:
        """The wrapper code plus the source of each kernel module whose
        benchmark harness (``get_args``/``call``) the module-level wrapper
        cut and moved there."""
        from torch._inductor.codecache import get_path

        keys = re.search(r"kernel_modules=(\[.*?\])", code)
        sources = [code]
        for key in ast.literal_eval(keys.group(1)) if keys else ():
            with open(get_path(key, "py")[2]) as f:
                sources.append(f.read())
        return "\n".join(sources)

    def _run_reduction(
        self,
        fn,
        M,
        K,
        N,
        test_config,
        *extra_configs,
        tol=0,
        dtype=torch.bfloat16,
        lo=-1,
        **patches,
    ):
        """Run fn on a (M, K) @ (K, N) GEMM and compare against eager. Returns the
        generated kernel names and the output code. Inputs are integers in
        [lo, lo + 2]; lo=1 makes every output positive, so a missing mask that
        reads zeros changes the result."""
        # Small integers make the GEMM and the sums exact, so results don't
        # depend on summation order and must match eager bitwise.
        a = torch.randint(lo, lo + 3, (M, K), device=GPU_TYPE).to(dtype)
        b = torch.randint(lo, lo + 3, (K, N), device=GPU_TYPE).to(dtype)
        with self._poison_outputs():
            actual, code = self._run_with_mm_config(
                fn, (a, b), test_config, *extra_configs, **patches
            )
        self.assertEqual(actual, fn(a, b), atol=tol, rtol=tol)
        return re.findall(r"def (triton_\w+)\(", code[0]), code[0]

    def _assert_row_fused(self, kernels, code, op="sum"):
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        FileCheck().check("tl.sum" if op != "amax" else "max2").run(code)

    ROW_OPS = {
        "sum": lambda a, b: (a @ b).float().sum(-1),
        # Every row is negative, so unmasked out-of-range columns would win.
        "amax": lambda a, b: ((a @ b) - 100).amax(-1),
        "mean": lambda a, b: (a @ b).float().mean(-1),
        "mean_and_sqmean": lambda a, b: (
            (c := (a @ b).float()).mean(-1),
            c.pow(2).mean(-1),
        ),
        "var_mean": lambda a, b: torch.var_mean((a @ b).float(), -1),
        "sum_and_out": lambda a, b: ((c := a @ b), c.float().sum(-1)),
        "extra_input": lambda a, b: ((a @ b).float() * b[0].float()).sum(-1),
    }

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", tuple(ROW_OPS))
    @parametrize("shape", ((1024, 128, 512), (1000, 120, 512)))
    @parametrize("variant", ("ws", "ws_tma_store", "ws_flatten"))
    @parametrize("epilogue_subtile", (1, 2, 4))
    def test_blackwell_mm_row_reduction_epilogue_fusion(
        self, op: str, shape: tuple[int, int, int], variant: str, epilogue_subtile: int
    ):
        if variant == "ws_flatten" and meta_ws_enabled():
            self.skipTest("meta WS doesn't support FLATTEN")
        if op == "var_mean" and epilogue_subtile > 1:
            self.skipTest(
                "covered by test_blackwell_mm_row_reduction_epilogue_not_fused"
            )
        M, N, K = shape
        test_config = BlackwellGPUGemmConfig(
            128,
            128,
            64,
            3,
            8,
            epilogue_subtile=epilogue_subtile,
            warp_specialize=True,
            flatten=variant == "ws_flatten",
        )
        kernels, code = self._run_reduction(
            self.ROW_OPS[op],
            M,
            K,
            N,
            test_config,
            # Divisions (mean, var) may be reassociated.
            tol=1e-5 if "mean" in op else 0,
            **{
                "triton.template_reduction_epilogue": True,
                "triton.enable_template_tma_store": variant == "ws_tma_store",
            },
        )
        self._assert_row_fused(kernels, code, op)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_reduction_epilogue_plain_stores_extra_outputs(self):
        """With a reduction epilogue, only the template's own output keeps its
        TMA store; extra outputs use tl.store."""
        kernels, code = self._run_reduction(
            lambda a, b: (
                (c := a @ b),
                *(c.float() * i for i in range(1, 4)),
                c.float().sum(-1),
            ),
            1024,
            512,
            128,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8, warp_specialize=True),
            **{
                "triton.template_reduction_epilogue": True,
                "triton.enable_template_tma_store": True,
            },
        )
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        self.assertEqual(len(re.findall(r"tma_descriptor\d+\.store\(", code)), 1)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_reduction_epilogue_fusion_fp16(self):
        kernels, code = self._run_reduction(
            self.ROW_OPS["sum_and_out"],
            1024,
            512,
            128,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            dtype=torch.float16,
            **{"triton.template_reduction_epilogue": True},
        )
        self._assert_row_fused(kernels, code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_reduction_epilogue_fusion_randn(self):
        fn = self.ROW_OPS["sum"]
        a = torch.randn(1024, 512, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(512, 128, device=GPU_TYPE, dtype=torch.bfloat16)
        with self._poison_outputs():
            actual, code = self._run_with_mm_config(
                fn,
                (a, b),
                BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
                **{"triton.template_reduction_epilogue": True},
            )
        torch.testing.assert_close(actual, fn(a, b), atol=1e-3, rtol=1e-5)
        self._assert_row_fused(re.findall(r"def (triton_\w+)\(", code[0]), code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_addmm_row_reduction_epilogue_fusion(self):
        M, K, N = 1024, 512, 128
        bias = torch.randint(-1, 2, (N,), device=GPU_TYPE).to(torch.bfloat16)

        def fn(a, b):
            return torch.addmm(bias, a, b).float().sum(-1)

        kernels, code = self._run_reduction(
            fn,
            M,
            K,
            N,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )
        self._assert_row_fused(kernels, code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_reduction_epilogue_subtile_spans_n(self):
        """With EPILOGUE_SUBTILE=2 and N == BLOCK_N // 2 (BLOCK_N is clamped to at
        least 16), the second subtile is fully masked and must not contribute."""
        if meta_ws_enabled():
            self.skipTest("meta WS has no valid choice for this GEMM")
        kernels, code = self._run_reduction(
            # Unmasked out-of-range columns would each add 1.
            lambda a, b: ((a @ b).float() + 1).sum(-1),
            1024,
            512,
            8,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8, epilogue_subtile=2),
            lo=1,
            **{"triton.template_reduction_epilogue": True},
        )
        self._assert_row_fused(kernels, code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize(
        "case",
        (
            "disabled",
            "unprofitable",
            "no_benchmark",
            "max_values",
            "argmax",
            "center_subtiled",
            "var_mean_subtiled",
            "data_partition",
            "two_ctas",
            "dynamic",
            "cpp_wrapper",
            "fp32",
            "var_mean_wide",
        ),
    )
    def test_blackwell_mm_row_reduction_epilogue_not_fused(self, case: str):
        """The reduction stays unfused when the flag is off, when the fused
        kernel benchmarks slower or isn't benchmarked, for arg reductions,
        subtiled outputs that read their reduction back or need a multi-output
        reduction, configs that can't host it (data partitioning, 2CTA),
        dynamic shapes, cpp_wrapper, fp32 outputs, and welford reductions of
        tiles narrower than N, which can't finish from partials."""
        fn = {
            "max_values": lambda a, b: (a @ b).float().max(-1).values,
            "argmax": lambda a, b: (a @ b).argmax(-1),
            "center_subtiled": lambda a, b: (
                (c := (a @ b).float()) - c.mean(-1, keepdim=True)
            ),
            "var_mean_subtiled": self.ROW_OPS["var_mean"],
            "var_mean_wide": lambda a, b: torch.var_mean((a @ b).float(), -1),
        }.get(case, self.ROW_OPS["sum"])
        M, K, N = 1024, 512, {"var_mean_wide": 256}.get(case, 128)
        test_config = BlackwellGPUGemmConfig(
            128,
            128,
            64,
            3,
            8,
            epilogue_subtile=2 if case.endswith("subtiled") else 1,
        )
        patches = {
            "benchmark_template_fusion": case != "no_benchmark",
            "triton.template_reduction_epilogue": case != "disabled",
            "triton.enable_template_tma_store": case == "two_ctas",
            "cpp_wrapper": case == "cpp_wrapper",
        }
        autows = {"data_partition": (2, False), "two_ctas": (1, True)}.get(
            case, (1, False)
        )
        if case in ("data_partition", "two_ctas") and not _use_template_autows():
            self.skipTest("needs template autoWS")
        if case == "two_ctas" and not has_two_ctas():
            self.skipTest("needs 2CTA support in Triton")
        if case == "fp32" and meta_ws_enabled():
            self.skipTest("fp32 GEMMs fail to compile under meta WS")
        if case == "cpp_wrapper":
            # The cpp wrapper can't embed the template's source.
            patches["triton.autotune_at_compile_time"] = True
        dtype = torch.float32 if case == "fp32" else torch.bfloat16
        a = torch.randint(-1, 2, (M, K), device=GPU_TYPE).to(dtype)
        b = torch.randint(-1, 2, (K, N), device=GPU_TYPE).to(dtype)
        actual, code = self._run_with_mm_config(
            fn,
            (a, b),
            test_config,
            autows=autows,
            dynamic=case == "dynamic" or None,
            # Otherwise fused kernels benchmark as free, so a perf rejection
            # can't hide a missing gate.
            fused_ms=1e6 if case == "unprofitable" else 0.0,
            **patches,
        )
        # Divisions (mean, var) may be reassociated.
        tol = (
            1e-5
            if case in ("center_subtiled", "var_mean_subtiled", "var_mean_wide")
            else 0
        )
        self.assertEqual(actual, fn(a, b), atol=tol, rtol=tol)
        kernels = re.findall(r"def (triton_\w+)\(", code[0])
        self.assertTrue(any(k.startswith("triton_tem") for k in kernels), kernels)
        self.assertTrue(
            any(k.startswith(("triton_per", "triton_red")) for k in kernels), kernels
        )

    COL_OPS = {
        "sum": lambda a, b: (a @ b).float().sum(0),
        # Unmasked out-of-range rows would each add 1.
        "plus1": lambda a, b: ((a @ b).float() + 1).sum(0),
        "amax": lambda a, b: ((a @ b) - 100).amax(0),
        "amin": lambda a, b: ((a @ b) + 100).amin(0),
        "mean": lambda a, b: (a @ b).float().mean(0),
        "center": lambda a, b: (c := (a @ b).float()) - c.sum(0),
        "sum_bf16": lambda a, b: (a @ b).sum(0),
        "sum_and_out": lambda a, b: ((c := a @ b), c.float().sum(0)),
        "relu_sum": lambda a, b: (a @ b).float().relu().sum(0, keepdim=True),
    }

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", tuple(COL_OPS))
    @parametrize("M", (1000, 1001, 4096))
    @parametrize("split", (False, True))
    @parametrize("epilogue_subtile", (1, 2, 4))
    def test_blackwell_mm_col_reduction_epilogue_fusion(
        self, op: str, M: int, split: bool, epilogue_subtile: int
    ):
        kernels, _ = self._run_reduction(
            self.COL_OPS[op],
            M,
            128,
            200,
            BlackwellGPUGemmConfig(
                128, 128, 64, 3, 8, epilogue_subtile=epilogue_subtile
            ),
            **{"triton.template_reduction_epilogue": True, "split_reductions": split},
        )
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        # Nodes reading the finished column results run after the template as
        # separate kernels.
        self.assertEqual(len(kernels), 2 if op in ("mean", "center") else 1, kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_col_reduction_epilogue_with_pointwise(self):
        """A pointwise node reading the output in place fuses next to a column
        reduction."""

        def fn(a, b):
            c = a @ b
            return c.relu(), c.float().sum(0)

        kernels, _ = self._run_reduction(
            fn,
            1024,
            128,
            256,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_col_reduction_epilogue_transposed_pointwise(self):
        """A pointwise node reading the output transposed doesn't fuse next to
        a column reduction, whose pass transposes the stored tile."""

        def fn(a, b):
            c = a @ b
            return c.t().contiguous(), c.float().sum(0)

        kernels, _ = self._run_reduction(
            fn,
            256,
            128,
            256,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertGreater(len(kernels), 1, kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("case", ("disabled", "argmax", "fp64", "dynamic", "data_partition"))
    def test_blackwell_mm_col_reduction_epilogue_not_fused(self, case: str):
        """The column reduction stays unfused when the flag is off, it is an
        arg reduction, its fp32 partials would lose precision, the shape is
        dynamic, or the config can't host it (data partitioning)."""
        fn = {
            "argmax": lambda a, b: (a @ b).float().argmax(0),
            "fp64": lambda a, b: (a @ b).double().sum(0),
        }.get(case, self.COL_OPS["sum"])
        if case == "data_partition" and not _use_template_autows():
            self.skipTest("needs template autoWS")
        test_config = BlackwellGPUGemmConfig(128, 128, 64, 3, 8)
        with torch._dynamo.config.patch(assume_static_by_default=case != "dynamic"):
            kernels, _ = self._run_reduction(
                fn,
                1024,
                128,
                256,
                test_config,
                autows=(2, False) if case == "data_partition" else (1, False),
                **{"triton.template_reduction_epilogue": case != "disabled"},
            )
        self.assertTrue(
            any(k.startswith(("triton_red", "triton_per")) for k in kernels), kernels
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize(
        "op",
        (
            "sum",
            "amax",
            "sum_and_out",
            "extra_input",
            "row_col",
            "mean",
            "mean_and_sqmean",
            "center",
        ),
    )
    @parametrize("M", (1000, 4096))
    @parametrize("N", (160, 200, 512))
    @parametrize("epilogue_subtile", (1, 2, 4))
    def test_blackwell_mm_row_reduction_epilogue_wide_n(
        self, op: str, M: int, N: int, epilogue_subtile: int
    ):
        """Row reductions of an output wider than the tile fuse as per-tile
        partials that the wrapper finishes. The last column tile ends in a fully
        masked subtile at N=160 with EPILOGUE_SUBTILE=2 and N=160 or 200 with 4."""
        fn = {
            "sum": lambda a, b: (a @ b).float().sum(1),
            # Every output is negative, so unmasked out-of-range columns,
            # which read zeros, would win.
            "amax": lambda a, b: (-(a @ b).float()).amax(1),
            "sum_and_out": lambda a, b: ((c := a @ b), c.float().sum(1)),
            "extra_input": lambda a, b: ((a @ b).float() * b[0].float()).sum(1),
            "row_col": lambda a, b: ((c := (a @ b).float()).sum(0), c.sum(1)),
            "mean": lambda a, b: (a @ b).float().mean(1),
            "mean_and_sqmean": lambda a, b: (
                (c := (a @ b).float()).mean(1),
                (c * c).mean(1),
            ),
            "center": lambda a, b: (c := (a @ b).float()) - c.sum(1, keepdim=True),
        }[op]
        kernels, _ = self._run_reduction(
            fn,
            M,
            128,
            N,
            BlackwellGPUGemmConfig(
                128, 128, 64, 3, 8, epilogue_subtile=epilogue_subtile
            ),
            tol=1e-5 if "mean" in op else 0,
            lo=1,
            **{"triton.template_reduction_epilogue": True},
        )
        # The reductions fuse; nodes reading their finished results run after.
        after = op in ("mean", "mean_and_sqmean", "center")
        self.assertEqual(
            [k.split("_fused")[0] for k in kernels],
            ["triton_tem", "triton_poi"] if after else ["triton_tem"],
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_reduction_epilogue_wide_n_unfused_kernels(self):
        """Nodes reading the finished partials of two row reductions run as two
        kernels after the template. Both read buffers whose last use is the
        template's node, which must stay alive until the second has run."""
        kernels, _ = self._run_reduction(
            lambda a, b: (
                (c := (a @ b).float()).mean(1),
                c * torch.rsqrt((c * c).mean(1, keepdim=True) + 1),
            ),
            1000,
            128,
            200,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            tol=1e-5,
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertEqual(
            [k.split("_fused")[0] for k in kernels],
            ["triton_tem", "triton_poi", "triton_poi"],
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_reduction_epilogue_wide_n_chained(self):
        """A row reduction reading the finished result of another can't run in
        the kernel that stores the first one's partials."""

        def fn(a, b):
            y = (a @ b).float()
            z = y - y.mean(-1, keepdim=True)
            return (z * z).sum(-1)

        kernels, _ = self._run_reduction(
            fn,
            1024,
            128,
            2048,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            tol=1e-5,
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertTrue(kernels[0].startswith("triton_tem"), kernels)
        self.assertTrue(
            any(k.startswith(("triton_per", "triton_red")) for k in kernels), kernels
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", ("sum", "gated", "amax_sum", "mean", "sum_and_out"))
    # At 4096x128 the split column reduction's group equals the output's; at
    # 65536 the two reductions form a mix-order reduction when unfused.
    @parametrize("M", (1000, 4096, 65536))
    @parametrize("epilogue_subtile", (1, 2, 4))
    def test_blackwell_mm_row_and_col_reduction_epilogue(
        self, op: str, M: int, epilogue_subtile: int
    ):
        """Row and column reductions of one output fuse into one epilogue,
        including at sizes where they would otherwise form a mix-order
        reduction."""
        source = torch.randint(-2, 3, (M, 128), device=GPU_TYPE).float()
        gate = torch.randint(0, 3, (M, 128), device=GPU_TYPE) / 2
        fn = {
            "sum": lambda a, b: ((c := (a @ b).float()).sum(0), c.sum(1)),
            "gated": lambda a, b: (
                (c := (a @ b).float() * source * gate * (1 - gate)).sum(0),
                c.sum(1),
            ),
            "amax_sum": lambda a, b: ((c := (a @ b).float()).sum(0), c.amax(1)),
            "mean": lambda a, b: ((c := (a @ b).float()).mean(0), c.mean(1)),
            "sum_and_out": lambda a, b: ((c := a @ b), c.float().sum(0), c.sum(1)),
        }[op]
        kernels, code = self._run_reduction(
            fn,
            M,
            128,
            128,
            BlackwellGPUGemmConfig(
                128, 128, 64, 3, 8, epilogue_subtile=epilogue_subtile
            ),
            tol=1e-5 if op == "mean" else 0,
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        # mean's column division reads the finished column sum.
        self.assertEqual(len(kernels), 2 if op == "mean" else 1, kernels)
        if op == "gated":
            # The column pass reuses the row pass's loads of source and gate.
            self.assertEqual(code.count("tl.load(in_ptr"), 2, code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_and_col_reduction_epilogue_masked_load(self):
        """The column pass doesn't reuse the row pass's masked loads: the row
        reduction reads source through a pad, which loads it only where the
        pad's mask holds, the column reduction reads all of it."""
        M = 1000
        source = torch.randint(-2, 3, (M, 128), device=GPU_TYPE).float()

        def fn(a, b):
            c = (a @ b).float()
            # A cat's branches also load source under masks, but their
            # indices don't match the column pass's, so a cat can't catch a
            # reused masked load.
            row = torch.nn.functional.pad(source[:, 64:], (64, 0))
            return (c * source).sum(0), (c * row).sum(1)

        kernels, _ = self._run_reduction(
            fn,
            M,
            128,
            128,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_and_col_reduction_epilogue_pw_reads_col(self):
        """The column reduction fuses even when a pointwise node reads its
        result. That node runs after the template, in the kernel the scheduler
        already fused it into with the row reduction."""

        def fn(a, b):
            c = (a @ b).float()
            s = c.sum(0)
            return s, c * s, c.sum(1)

        kernels, _ = self._run_reduction(
            fn,
            1024,
            128,
            128,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertEqual(len(kernels), 2, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        self.assertIn("sum", kernels[0])
        self.assertTrue(kernels[1].startswith("triton_per"), kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("axis", (0, 1))
    def test_blackwell_mm_reduction_epilogue_fused_meta_ws_subtiled(self, axis: int):
        """Subtiled reduction epilogues fuse under Meta automatic warp
        specialization (facebookexperimental/triton#3803 keeps every subtile
        read after the accumulator-ready wait)."""
        if not meta_ws_enabled():
            self.skipTest("needs Meta Triton autoWS enabled")
        kernels, _ = self._run_reduction(
            lambda a, b: ((c := a @ b), c.float().sum(axis)),
            1024,
            128,
            256,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8, epilogue_subtile=2),
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertFalse(
            any(k.startswith(("triton_red", "triton_per")) for k in kernels), kernels
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("epilogue_subtile", (2, 4))
    def test_blackwell_mm_reduction_epilogue_meta_ws_subtiled_persistent(
        self, epilogue_subtile: int
    ):
        """More tiles than SMs, so each CTA's next MMA reuses the accumulator: a
        subtile read moved past the accumulator release would see the next tile."""
        if not meta_ws_enabled():
            self.skipTest("needs Meta Triton autoWS enabled")
        for _ in range(3):
            kernels, code = self._run_reduction(
                lambda a, b: ((c := a @ b), c.float().sum(-1), c.float().sum(0)),
                32768,
                128,
                256,
                BlackwellGPUGemmConfig(
                    128, 128, 64, 4, 8, epilogue_subtile=epilogue_subtile
                ),
                **{"triton.template_reduction_epilogue": True},
            )
            # The template kernel's body, as a module-level def or a source string.
            template_src = re.search(
                r"^def triton_tem_\w+\(.*?(?=^\S|\Z)", code, re.DOTALL | re.MULTILINE
            )
            self.assertIsNotNone(template_src, code)
            self.assertIn("tl.sum", template_src.group(0))

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", ("col_reads_row", "row_reads_col"))
    def test_blackwell_mm_row_and_col_reduction_epilogue_not_fused(self, op: str):
        """A column reduction may not read a row result, and no reduction may
        read a column result, since column results finish after the kernel."""
        fn = {
            "col_reads_row": lambda a, b: (
                (d := (c := (a @ b).float()) - c.sum(1, keepdim=True)),
                d.sum(0),
            ),
            "row_reads_col": lambda a, b: (
                (s := (c := (a @ b).float()).sum(0)),
                (c * s).sum(1),
            ),
        }[op]
        kernels, _ = self._run_reduction(
            fn,
            1024,
            128,
            128,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertGreater(len(kernels), 1, kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", ("col", "row_wide"))
    def test_blackwell_mm_reduction_epilogue_benchmark_times_finish(self, op: str):
        """The benchmark harness of a kernel with reduction partials also runs
        the wrapper's finish, so epilogue benchmarking pays for it."""
        fn, n = {
            "col": (self.COL_OPS["sum"], 128),
            "row_wide": (lambda a, b: (a @ b).float().sum(1), 256),
        }[op]
        _, code = self._run_reduction(
            fn,
            1024,
            128,
            n,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True, "benchmark_kernel": True},
        )
        size = 1024 if op == "row_wide" else n
        finish = re.findall(
            r"\.run\(\*args, stream=\w+\)\n\s+args\[\d+\]\[\d+:\d+\]"
            rf"\.view\(torch\.float32\)\.view\(\d+, {size}\)\.sum\(dim=0\)",
            self._with_kernel_harnesses(code),
        )
        self.assertEqual(len(finish), 1, code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_reduction_epilogue_benchmark_casts_finish(self):
        """For a bf16 column reduction, the benchmark's finish also pays for the
        wrapper's cast to the output dtype, as the unfused side does."""
        _, code = self._run_reduction(
            self.COL_OPS["sum_bf16"],
            1024,
            128,
            128,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True, "benchmark_kernel": True},
        )
        finish = re.findall(
            r"\.run\(\*args, stream=\w+\)\n\s+args\[\d+\]\[\d+:\d+\]"
            r"\.view\(torch\.float32\)\.view\(\d+, 128\)\.sum\(dim=0\)"
            r"\.to\(torch\.bfloat16\)",
            self._with_kernel_harnesses(code),
        )
        self.assertEqual(len(finish), 1, code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_reduction_epilogue_benchmark_mix_order(self):
        """Epilogue benchmarking times the real mix-order reduction kernel, and
        restores its nodes for codegen afterwards."""
        from torch._inductor.codegen.simd import SIMDScheduling

        timed = []
        bench = SIMDScheduling.benchmark_mix_order_reduction

        def spy(self, node):
            timed.append(bench(self, node))
            return timed[-1]

        with mock.patch.object(SIMDScheduling, "benchmark_mix_order_reduction", spy):
            self._run_reduction(
                lambda a, b: ((y := (a @ b).float()).sum(0), y.sum(1)),
                65536,
                128,
                128,
                BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
                BlackwellGPUGemmConfig(128, 64, 64, 3, 8),
                **{
                    "triton.template_reduction_epilogue": True,
                    "benchmark_template_fusion": True,
                },
            )
        self.assertTrue(timed and None not in timed, timed)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_reduction_epilogue_benchmark_unsplit_mix_order(self):
        """Benchmarking a mix-order reduction whose column reduction isn't
        split leaves that reduction's loop body unchanged for codegen."""
        self._run_reduction(
            lambda a, b: ((y := (a @ b).float()).sum(0), y.sum(1)),
            65536,
            128,
            128,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{
                "triton.template_reduction_epilogue": True,
                "split_reductions": False,
            },
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_reduction_epilogue_benchmark_wide_mix_order(self):
        """An output wide enough that its column reduction isn't split forms
        an unsplit mix-order reduction under the default config."""
        self._run_reduction(
            lambda a, b: ((y := (a @ b).float()).sum(0), y.sum(1)),
            8192,
            128,
            16384,
            BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_reduction_epilogue_benchmark_deferred_epilogue(self):
        """Epilogue benchmarking also times the nodes run after the finish of
        the reduction partials."""
        from torch._inductor.scheduler import Scheduler

        benchmarked = []
        bench = Scheduler.benchmark_fused_nodes

        def spy(self, nodes, *args, **kwargs):
            benchmarked.append(nodes)
            return bench(self, nodes, *args, **kwargs)

        with mock.patch.object(Scheduler, "benchmark_fused_nodes", spy):
            self._run_reduction(
                lambda a, b: ((y := (a @ b).float()).mean(1), (y * y).mean(1)),
                1024,
                128,
                256,
                BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
                BlackwellGPUGemmConfig(128, 64, 64, 3, 8),
                tol=1e-5,
                unfused_ms=None,
                **{
                    "triton.template_reduction_epilogue": True,
                    "benchmark_template_fusion": True,
                },
            )
        deferred = [n for n in benchmarked if not any(x.is_reduction() for x in n)]
        self.assertEqual(len(deferred), 1, benchmarked)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", ("row", "col"))
    def test_blackwell_mm_reduction_epilogue_transposed_read(self, op: str):
        """A square output read transposed by a pointwise node must not fuse,
        since epilogue loads take the tile element at the store position."""
        fn = {
            "row": lambda a, b: ((p := (a @ b).t().contiguous()), p.float().sum(1)),
            "col": lambda a, b: ((c := a @ b).t().contiguous(), c.float().sum(0)),
        }[op]
        kernels, _ = self._run_reduction(
            fn,
            256,
            128,
            256,
            BlackwellGPUGemmConfig(128, 256, 64, 3, 8),
            **{"triton.template_reduction_epilogue": True},
        )
        self.assertFalse(
            any(k.startswith("triton_tem") and "clone" in k for k in kernels), kernels
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_mm_row_reduction_epilogue_skips_misfit_tile(self):
        """The fusion benchmark skips choices whose tile can't host the
        reduction epilogue, even when they rank first, and fuses into one that
        can. Reading a row reduction back doesn't fit subtiled outputs."""
        orig_choice_timings = MultiTemplateBuffer.choice_timings
        orig_tile_fits = simd.tile_fits_reduction_epilogue
        rejected_tiles = []

        def choice_timings(self, hint_override=None):
            # Rank the subtiled choices first.
            timings = orig_choice_timings(self, hint_override)
            subtiled = [c for c in timings if (c.output_tile or (0, 0, 1))[2] > 1]
            return dict.fromkeys([*subtiled, *timings], 1.0)

        def tile_fits(tile, template, epilogue_nodes):
            fits = orig_tile_fits(tile, template, epilogue_nodes)
            if not fits:
                rejected_tiles.append(tile)
            return fits

        with (
            mock.patch.object(MultiTemplateBuffer, "choice_timings", choice_timings),
            mock.patch.object(simd, "tile_fits_reduction_epilogue", tile_fits),
        ):
            kernels, code = self._run_reduction(
                lambda a, b: (c := (a @ b).float()) - c.mean(-1, keepdim=True),
                1024,
                512,
                128,
                BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
                BlackwellGPUGemmConfig(128, 128, 64, 3, 8, epilogue_subtile=2),
                tol=1e-5,
                **{"triton.template_reduction_epilogue": True},
            )
        self._assert_row_fused(kernels, code)
        self.assertTrue(rejected_tiles)
        self.assertTrue(
            all(tile is not None and tile[2] > 1 for tile in rejected_tiles),
            rejected_tiles,
        )

    def _run_stale_pending_fusion(self, compile_mode: str, **patches):
        """Compile a GEMM whose epilogue fusion with a pointwise node is
        benchmarked before that node fuses with a row reduction. The benchmark
        picks a subtiled choice, whose tile can't hold var_mean."""
        if compile_mode == "async" and not AsyncCompile.wait_process_pool_ready():
            self.skipTest("compile worker pool unavailable")
        orig_choice_timings = MultiTemplateBuffer.choice_timings

        def choice_timings(self, hint_override=None):
            timings = orig_choice_timings(self, hint_override)
            subtiled = [c for c in timings if (c.output_tile or (0, 0, 1))[2] > 1]
            return dict.fromkeys([*subtiled, *timings], 1.0)

        with (
            mock.patch.object(MultiTemplateBuffer, "choice_timings", choice_timings),
            mock.patch.object(AsyncCompile, "use_process_pool", return_value=False)
            if compile_mode == "sync"
            else contextlib.nullcontext(),
        ):
            return self._run_reduction(
                lambda a, b: ((c := a @ b) * 2, torch.var_mean(c.float(), -1)),
                1024,
                512,
                128,
                BlackwellGPUGemmConfig(128, 128, 64, 3, 8),
                BlackwellGPUGemmConfig(128, 128, 64, 3, 8, epilogue_subtile=2),
                tol=1e-5,
                **{
                    "triton.template_reduction_epilogue": True,
                    "epilogue_fusion_first": True,
                    **patches,
                },
            )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("compile_mode", ("sync", "async"))
    def test_blackwell_mm_reduction_epilogue_stale_pending_fusion(
        self, compile_mode: str
    ):
        """A pending epilogue fusion whose pointwise node has since fused with a
        row reduction isn't applied, and the next fusion round benchmarks the
        template with both."""
        kernels, code = self._run_stale_pending_fusion(compile_mode)
        self._assert_row_fused(kernels, code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("compile_mode", ("sync", "async"))
    def test_blackwell_mm_reduction_epilogue_stale_pending_fusion_memory_guard(
        self, compile_mode: str
    ):
        """The peak-memory fusion guard still rejects the template fusion that
        the next round proposes after a stale pending fusion is skipped."""
        orig_check = Scheduler._can_fuse_peak_memory_check
        orig_is_stale = Scheduler._is_stale_pending_fusion
        stale, rejected = [], []

        def check(self, state, node1, node2):
            # Treat template fusions with the fused pointwise and reduction
            # nodes as exceeding the allowed peak increase.
            nodes = node2.get_nodes()
            if node1.is_template() and len(nodes) > 1:
                if any(n.is_reduction() for n in nodes):
                    rejected.append(node2)
                    return False, None
            return orig_check(self, state, node1, node2)

        def is_stale(self, pending_fusion):
            stale.append(orig_is_stale(self, pending_fusion))
            return stale[-1]

        with (
            mock.patch.object(Scheduler, "_can_fuse_peak_memory_check", check),
            mock.patch.object(Scheduler, "_is_stale_pending_fusion", is_stale),
        ):
            kernels, _ = self._run_stale_pending_fusion(
                compile_mode, fusion_memory_timeline_peak_memory_increase_gb=1000.0
            )
        self.assertIn(True, stale)
        self.assertTrue(rejected)
        self.assertTrue(any(k.startswith("triton_tem") for k in kernels), kernels)
        self.assertTrue(
            any(k.startswith(("triton_per", "triton_red")) for k in kernels), kernels
        )

    def _assert_template_row_sum_fused(
        self, fn, args, template, op_name, test_config, expected
    ):
        """Compile fn with test_config, and a twin so epilogues are benchmarked,
        as the only choices of template. Fused epilogue benchmarks report 0 ms."""
        heuristic = get_template_heuristic(template.uid, GPU_TYPE, op_name)
        twin = dataclasses.replace(test_config, num_stages=test_config.num_stages - 1)
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "triton.enable_persistent_tma_matmul": True,
                    "test_configs.autotune_choice_name_regex": template.name,
                    "benchmark_template_fusion": True,
                    "triton.template_reduction_epilogue": True,
                }
            ),
            mock.patch.object(heuristic, "mm_configs", [test_config, twin]),
            mock.patch.object(
                Scheduler, "benchmark_codegened_module", return_value=(0.0, "")
            ),
            self._poison_outputs(),
        ):
            actual, code = run_and_get_code(torch.compile(fn), *args)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        kernels = re.findall(r"def (triton_\w+)\(", code[0])
        # A scale layout fixup may run as a separate pointwise kernel.
        self.assertFalse(
            any(k.startswith(("triton_per", "triton_red")) for k in kernels), kernels
        )
        self.assertTrue(
            any(k.startswith("triton_tem_fused") and "sum" in k for k in kernels),
            kernels,
        )

    @unittest.skipIf(not has_triton_tma_device(), "Need device-side TMA support")
    def test_persistent_tma_mm_row_reduction_epilogue_fusion(self):
        a = torch.randint(-1, 2, (1024, 512), device=GPU_TYPE).to(torch.bfloat16)
        b = torch.randint(-1, 2, (512, 128), device=GPU_TYPE).to(torch.bfloat16)
        fn = self.ROW_OPS["sum"]
        # On datacenter Blackwell the Blackwell template replaces this one.
        with mock.patch.object(
            torch._inductor.kernel.mm,
            "use_triton_blackwell_tma_template",
            return_value=False,
        ):
            self._assert_template_row_sum_fused(
                fn,
                (a, b),
                persistent_tma_mm_template,
                "mm",
                GemmConfig(128, 128, 64, 3, 8),
                fn(a, b),
            )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_scaled_mm_epilogue_scaling_row_reduction_epilogue_fusion(self):
        M, K, N = 1024, 512, 128
        a = torch.randint(-1, 2, (M, K), device=GPU_TYPE).to(torch.float8_e4m3fn)
        b = torch.randint(-1, 2, (N, K), device=GPU_TYPE).to(torch.float8_e4m3fn).t()
        # Power-of-two scales keep the GEMM exact, so it matches eager bitwise.
        scale_a = 2.0 ** torch.randint(0, 2, (M, 1), device=GPU_TYPE)
        scale_b = 2.0 ** torch.randint(0, 2, (1, N), device=GPU_TYPE)

        def fn(a, b, scale_a, scale_b):
            c = torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)
            return c.float().sum(-1)

        args = (a, b, scale_a, scale_b)
        self._assert_template_row_sum_fused(
            fn,
            args,
            scaled_mm_device_tma_epilogue_scaling_template,
            "scaled_mm",
            GemmConfig(128, 128, 128, 3, 8),
            fn(*args),
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_scaled_mm_main_loop_scaling_row_reduction_epilogue_fusion(self):
        M, K, N = 1024, 256, 128
        a = torch.randint(-1, 2, (M, K), device=GPU_TYPE).to(torch.float8_e4m3fn)
        b = torch.randint(-1, 2, (N, K), device=GPU_TYPE).to(torch.float8_e4m3fn)
        # 1x128 scales for a (stored outer-dim-major) and 128x128 scales for b.
        scale_a = 2.0 ** torch.randint(0, 2, (K // 128, M), device=GPU_TYPE).t()
        scale_b = 2.0 ** torch.randint(0, 2, (N // 128, K // 128), device=GPU_TYPE)

        def fn(a, b, scale_a, scale_b):
            c = scaled_mm(
                a,
                b,
                scale_a,
                ScalingType.BlockWise1x128,
                scale_b,
                ScalingType.BlockWise128x128,
                output_dtype=torch.bfloat16,
            )
            return c.float().sum(-1)

        # Eager blockwise scaling needs SM90 cuBLAS, so compare against an exact
        # reference: integer inputs and power-of-two scales make the GEMM exact
        # in fp64 and fp32, and its bf16 rounding and row sums then match.
        scale_a_full = scale_a.double().repeat_interleave(128, 1)
        scale_b_full = scale_b.double().repeat_interleave(128, 0)
        scale_b_full = scale_b_full.repeat_interleave(128, 1)
        c = (a.double() * scale_a_full) @ (b.double() * scale_b_full).t()
        self._assert_template_row_sum_fused(
            fn,
            (a, b.t(), scale_a, scale_b),
            scaled_mm_device_tma_main_loop_scaling_template,
            "scaled_mm",
            GemmConfig(128, 128, 128, 3, 8),
            c.bfloat16().float().sum(-1),
        )


@instantiate_parametrized_tests
class TestBlackwellBMMReductionEpilogue(TestCase):
    """Reduction epilogues of the Blackwell persistent TMA BMM template."""

    def _run_bmm_reduction(
        self, fn, B, M, K, N, test_config, *, extra_input=False, tol=0, **patches
    ):
        """Run fn on a (B, M, K) @ (B, K, N) bmm with only test_config and its
        num_stages - 1 twin as choices, keeping every fusion the gates allow, and
        compare against eager. extra_input adds a third input, of shape
        (B, M, N) or the given shape. Returns the generated kernel names and the
        code."""
        # Small integers make the bmm and the sums exact, so results don't
        # depend on summation order and must match eager bitwise.
        a = torch.randint(-1, 2, (B, M, K), device=GPU_TYPE).to(torch.bfloat16)
        b = torch.randint(-1, 2, (B, K, N), device=GPU_TYPE).to(torch.bfloat16)
        args = (a, b)
        if extra_input:
            shape = (B, M, N) if extra_input is True else extra_input
            args += (torch.randint(-2, 3, shape, device=GPU_TYPE).float(),)
        configs = [
            test_config,
            # Epilogues are only benchmarked, and so reductions only fused,
            # when autotuning has more than one choice.
            dataclasses.replace(test_config, num_stages=test_config.num_stages - 1),
        ]
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "blackwell_bmm",
                    "benchmark_template_fusion": True,
                    "triton.template_reduction_epilogue": True,
                    **patches,
                }
            ),
            mock.patch.object(
                CUDABlackwellBMMTemplateConfigHeuristic, "bmm_configs", configs
            ),
            mock.patch.object(
                Scheduler, "benchmark_codegened_module", return_value=(0.0, "")
            ),
            # Unfused kernels take 1 ms, so the nodes that run after a fused
            # kernel cost no more than the unfused epilogue they replace (real
            # timings make those fusions flaky).
            mock.patch.object(
                Scheduler, "benchmark_fused_nodes", return_value=(1.0, "")
            ),
            TestBlackwellTMALoadFusion._poison_outputs(),
        ):
            actual, code = run_and_get_code(torch.compile(fn), *args)
        self.assertEqual(actual, fn(*args), atol=tol, rtol=tol)
        return re.findall(r"def (triton_\w+)\(", code[0]), code[0]

    BMM_OPS = {
        "row_sum": lambda a, b: (a @ b).float().sum(-1),
        # Out-of-range lanes hold f(0), here 1, so each one a mask lets through
        # changes the sum.
        "row_sum_offset": lambda a, b: ((a @ b).float() + 1).sum(-1),
        # Every value is negative, so masked lanes that hold 0 rather than the
        # max's identity would win.
        "row_amax": lambda a, b: ((a @ b) - 100).amax(-1),
        "row_mean": lambda a, b: (a @ b).float().mean(-1),
        "row_sum_and_out": lambda a, b: ((c := a @ b), c.float().sum(-1)),
        # The pointwise output and the row sum share c.float().
        "row_sum_shared_with_out": lambda a, b: (
            (c := (a @ b).float()) + 1,
            (c * 2).sum(-1),
        ),
        "col_sum": lambda a, b: (a @ b).float().sum((0, 1)),
        "col_sum_offset": lambda a, b: ((a @ b).float() + 1).sum((0, 1)),
        "batch_col_sum": lambda a, b: (a @ b).float().sum(1),
        "batch_col_sum_offset": lambda a, b: ((a @ b).float() + 1).sum(1),
        "batch_col_amax": lambda a, b: ((a @ b) - 100).amax(1),
        "batch_col_sum_extra_input": lambda a, b, w: ((a @ b).float() * w).sum(1),
        "batch_sum": lambda a, b: (a @ b).float().sum(0),
        "batch_sum_bf16": lambda a, b: (a @ b).sum(0),
        "batch_sum_and_out": lambda a, b: ((c := a @ b), c.float().sum(0)),
        "batch_sum_extra_input": lambda a, b, w: ((a @ b).float() * w).sum(0),
        # RMSNorm backward: batch and row sums of one product.
        "batch_and_row_sum": lambda a, b, w: (
            (c := (a @ b).float() * w).sum(0),
            c.sum(-1),
        ),
        "all_axes": lambda a, b, w: (
            (c := (a @ b).float() * w).sum(0),
            c.sum(1),
            c.sum(-1),
            c.sum((0, 1)),
        ),
        # Reductions over each batch's (M, N) matrix.
        "matrix_sum": lambda a, b: (a @ b).float().sum((1, 2)),
        "matrix_sum_offset": lambda a, b: ((a @ b).float() + 1).sum((1, 2)),
        "matrix_sum_bf16": lambda a, b: (a @ b).sum((1, 2)),
        "matrix_amax": lambda a, b: ((a @ b) - 100).amax((1, 2)),
        "matrix_mean": lambda a, b: (a @ b).float().mean((1, 2)),
        "matrix_sum_flat_view": lambda a, b: (a @ b).float().flatten(1).sum(1),
        "matrix_sum_and_out": lambda a, b: ((c := a @ b), c.float().sum((1, 2))),
        "matrix_sum_extra_input": lambda a, b, w: ((a @ b).float() * w).sum((1, 2)),
        "matrix_and_row_sum": lambda a, b, w: (
            (c := (a @ b).float() * w).sum((1, 2)),
            c.sum(-1),
        ),
    }

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", tuple(BMM_OPS))
    # M % BLOCK_M != 0, so tiles hold rows past each batch's M; N tails; one
    # and two tiles per batch; more tiles than SMs.
    @parametrize("shape", ((300, 96, 64, 128), (64, 176, 64, 80)))
    @parametrize("tma_store", (False, True))
    def test_blackwell_bmm_reduction_epilogue_fusion(
        self, op: str, shape: tuple[int, int, int, int], tma_store: bool
    ):
        fn = self.BMM_OPS[op]
        kernels, code = self._run_bmm_reduction(
            fn,
            *shape,
            BlackwellBMMConfig(128, 128, 64, 3, 8),
            extra_input=fn.__code__.co_argcount == 3,
            # Divisions (mean) may be reassociated.
            tol=1e-5 if "mean" in op else 0,
            **{"triton.enable_template_tma_store": tma_store},
        )
        # The division of a matrix mean reads the per-batch result, so it runs
        # after the kernel.
        self.assertEqual(len(kernels), 2 if op == "matrix_mean" else 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        self.assertTrue(all(k.startswith("triton_poi") for k in kernels[1:]), kernels)
        if "batch_sum" in op or op in ("batch_and_row_sum", "all_axes"):
            # Each program stores its accumulator once, on its last tile.
            FileCheck().check("_batch_acc0 = tl.zeros(").check("for tile_id").check(
                "_batch_acc0 = _batch_acc0 +"
            ).check("tile_id + NUM_SMS >= num_tiles").run(code)
        if op.startswith("matrix"):
            FileCheck().check("for tile_id").check("tl.where(").run(code)
            # A matrix spanning two tiles reduces from per-tile partials, as
            # does one whose result is read only over the batches.
            partials = shape[1] > 128 or op == "matrix_mean"
            self.assertEqual("tl.pointer_type(tl.float32)" in code, partials)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("tma_store", (False, True))
    # A per-element input, and ones broadcast over the batches or the columns.
    @parametrize("w_shape", ((300, 96, 128), (96, 128), (300, 96, 1)))
    def test_blackwell_bmm_row_pass_reuses_epilogue_loads(
        self, tma_store: bool, w_shape: tuple[int, ...]
    ):
        """The row pass reuses the extra input that the pointwise epilogue
        before it loaded, rather than reload it into a second register tile."""
        kernels, code = self._run_bmm_reduction(
            lambda a, b, w: ((x := (a @ b).float() * w), (x * w).sum(-1)),
            300,
            96,
            64,
            128,
            BlackwellBMMConfig(128, 128, 64, 3, 8),
            extra_input=w_shape,
            **{"triton.enable_template_tma_store": tma_store},
        )
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        loads = re.findall(r"tl\.load\((in_ptr\d+) ", code)
        self.assertEqual(len(loads), len(set(loads)), loads)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("tma_store", (False, True))
    def test_blackwell_bmm_row_pass_skips_masked_epilogue_loads(self, tma_store: bool):
        """A load under a mask (here a pad's) is 0 outside it, so the row pass
        must load the input again rather than reuse it."""
        kernels, _ = self._run_bmm_reduction(
            lambda a, b, w: (
                (c := (a @ b).float()) + torch.nn.functional.pad(w[..., :-16], (0, 16)),
                (c * w).sum(-1),
            ),
            300,
            96,
            64,
            128,
            BlackwellBMMConfig(128, 128, 64, 3, 8),
            extra_input=True,
            **{"triton.enable_template_tma_store": tma_store},
        )
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_bmm_reduction_epilogue_plain_stores_extra_outputs(self):
        """With a reduction epilogue, only the template's own output keeps its
        TMA store: one staged fp32 128x128 tile per extra output would overflow
        shared memory (an RMSNorm backward epilogue can store up to 17)."""

        def fn(a, b, w):
            c = a @ b
            x = c.float() * w
            return (c, *(x * i for i in range(1, 7)), x.sum(-1))

        kernels, code = self._run_bmm_reduction(
            fn,
            300,
            128,
            64,
            128,
            BlackwellBMMConfig(128, 128, 64, 3, 8),
            extra_input=True,
            **{"triton.enable_template_tma_store": True},
        )
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        self.assertEqual(len(re.findall(r"tma_descriptor\d+\.store\(", code)), 1)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize(
        "shape",
        (
            # One and two tiles per batch, M and N tails, many batches.
            (5139, 128, 128, 64),
            (5139, 72, 128, 96),
            (5139, 176, 128, 128),
            (300, 8, 128, 128),
        ),
    )
    @parametrize("tma_store", (False, True))
    @parametrize("direction", ("fwd", "bwd"))
    def test_blackwell_bmm_matrix_reduction_two_pass(
        self, shape: tuple[int, int, int, int], tma_store: bool, direction: str
    ):
        """RMSNorm over each batch's matrix. When one tile holds a batch's
        matrix, the statistic and the pass that reads it back fuse into one
        kernel, so the bmm output is never stored; otherwise the statistic fuses
        as per-tile partials and that pass runs after the kernel."""
        B, M, _, N = shape

        def fwd(a, b, w):
            c = (a @ b).float()
            return (c * torch.rsqrt(c.pow(2).mean((1, 2), keepdim=True) + 1e-5)) * w

        def bwd(a, b, w):
            c = (a @ b).float()
            return c - w * (c * w).sum((1, 2), keepdim=True) / (M * N)

        kernels, code = self._run_bmm_reduction(
            fwd if direction == "fwd" else bwd,
            *shape,
            BlackwellBMMConfig(128, 128, 64, 3, 8),
            extra_input=True,
            # rsqrt and the division may round differently from eager.
            tol=1e-5,
            **{"triton.enable_template_tma_store": tma_store},
        )
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)
        self.assertFalse(
            any(k.startswith(("triton_red", "triton_per")) for k in kernels), kernels
        )
        if M <= 128 and N <= 128:
            self.assertEqual(len(kernels), 1, kernels)
            # The bmm output is dead: no buffer of its (B, M, N) shape is
            # allocated.
            self.assertNotIn(
                f"(({B}, {M}, {N}), ({M * N}, {N}, 1), torch.bfloat16)", code
            )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize(
        "case",
        (
            # bmm -> RMSNorm backward.
            ("batch_and_row_sum", (5247, 128, 72, 128)),
            ("row_sum", (5247, 128, 72, 128)),
            ("batch_sum", (5247, 256, 16, 80)),
            # 3 tiles per batch with a program count they divide, and 4.
            ("batch_sum", (40, 128, 64, 384)),
            ("batch_sum", (300, 256, 64, 256)),
            ("row_sum", (5247, 256, 16, 80)),
            ("batch_col_sum", (5247, 80, 16, 256)),
        ),
    )
    def test_blackwell_bmm_reduction_epilogue_large_batch(self, case):
        op, shape = case
        fn = self.BMM_OPS[op]
        kernels, _ = self._run_bmm_reduction(
            fn,
            *shape,
            BlackwellBMMConfig(128, 128, 64, 3, 8),
            extra_input=fn.__code__.co_argcount == 3,
            **{"triton.enable_template_tma_store": True},
        )
        self.assertEqual(len(kernels), 1, kernels)
        self.assertTrue(kernels[0].startswith("triton_tem_fused"), kernels)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize(
        "case", ("tiles_per_batch", "subtiled", "batch_amax", "batch_and_n_sum", "sum")
    )
    def test_blackwell_bmm_batch_reduction_epilogue_not_fused(self, case: str):
        """The batch sum stays unfused when programs of the persistent loop
        would see different tiles of a batch (3 tiles per batch don't divide
        the SM count), or with epilogue subtiles. So do batch maxes and
        reductions over the batch and N dims, or over all dims."""
        if case == "tiles_per_batch" and get_num_sms() % 3 == 0:
            self.skipTest("needs an SM count that 3 doesn't divide")
        if case == "subtiled" and meta_ws_enabled():
            self.skipTest("meta WS doesn't fuse reductions over subtiles")
        ops = {
            "batch_amax": lambda a, b: ((a @ b) - 100).amax(0),
            "batch_and_n_sum": lambda a, b: (a @ b).float().sum((0, 2)),
            "sum": lambda a, b: (a @ b).float().sum(),
        }
        kernels, _ = self._run_bmm_reduction(
            ops.get(case, self.BMM_OPS["batch_sum"]),
            # 3 tiles per batch and more tiles than SMs, or 1 tile per batch.
            *((512, 128, 64, 384) if case == "tiles_per_batch" else (300, 96, 64, 128)),
            BlackwellBMMConfig(
                128, 128, 64, 3, 8, epilogue_subtile=2 if case == "subtiled" else 1
            ),
        )
        self.assertTrue(
            any(k.startswith(("triton_red", "triton_per")) for k in kernels), kernels
        )


@instantiate_parametrized_tests
class TestBlackwellExhaustiveConfigs(TestCase):
    """Tests for exhaustive config generation for Blackwell templates."""

    # Expected valid values for each parameter based on _generate_exhaustive_configs
    VALID_BLOCK_SIZES = {32, 64, 128, 256}
    VALID_GROUP_M = {8}
    VALID_NUM_STAGES = {2, 3, 4, 5, 6}
    VALID_NUM_WARPS = {4, 8}
    VALID_EPILOGUE_SUBTILE = {1, 2, 4}

    @parametrize(
        "heuristic_cls",
        (
            CUDABlackwellPersistentTMATemplateConfigHeuristic,
            CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic,
        ),
    )
    def test_exhaustive_configs_parameter_variety(self, heuristic_cls):
        """Test that exhaustive configs have variety in all expected parameters."""
        heuristic = heuristic_cls()
        configs = heuristic.exhaustive_configs

        # Ensure we have at least 128 configs
        self.assertGreaterEqual(len(configs), 128)

        # Collect all unique values for each parameter
        block_m_values = set()
        block_n_values = set()
        block_k_values = set()
        group_m_values = set()
        num_stages_values = set()
        num_warps_values = set()
        epilogue_subtile_values = set()

        for cfg in configs:
            self.assertIsInstance(cfg, BlackwellGPUGemmConfig)
            block_m_values.add(cfg.block_m)
            block_n_values.add(cfg.block_n)
            block_k_values.add(cfg.block_k)
            group_m_values.add(cfg.group_m)
            num_stages_values.add(cfg.num_stages)
            num_warps_values.add(cfg.num_warps)
            epilogue_subtile_values.add(cfg.epilogue_subtile)

        # Verify multiple values exist for each parameter via set membership
        self.assertEqual(block_m_values, self.VALID_BLOCK_SIZES)
        self.assertEqual(block_n_values, self.VALID_BLOCK_SIZES)
        self.assertEqual(block_k_values, self.VALID_BLOCK_SIZES)
        self.assertEqual(group_m_values, self.VALID_GROUP_M)
        self.assertEqual(num_stages_values, self.VALID_NUM_STAGES)
        self.assertEqual(num_warps_values, self.VALID_NUM_WARPS)
        self.assertEqual(epilogue_subtile_values, self.VALID_EPILOGUE_SUBTILE)

    def test_scaled_blackwell_tma_uses_scaled_persistent_configs(self):
        """Verify scaled Blackwell TMA heuristic uses blackwell_scaled_persistent_mm_configs."""
        heuristic = CUDAScaledBlackwellTMATemplateConfigHeuristic()

        for cfg in heuristic.mm_configs:
            self.assertIsInstance(cfg, BlackwellGPUGemmConfig)

        expected_configs = heuristic.blackwell_scaled_persistent_mm_configs
        self.assertEqual(len(heuristic.mm_configs), len(expected_configs))
        for actual, expected in zip(heuristic.mm_configs, expected_configs):
            self.assertEqual(actual.block_m, expected.block_m)
            self.assertEqual(actual.block_n, expected.block_n)
            self.assertEqual(actual.block_k, expected.block_k)
            self.assertEqual(actual.num_stages, expected.num_stages)
            self.assertEqual(actual.num_warps, expected.num_warps)

        addmm_configs = heuristic.blackwell_persistent_addmm_configs
        self.assertGreater(
            len(heuristic.mm_configs),
            len(addmm_configs),
            "Scaled TMA should use the larger scaled_persistent list, not the small addmm list",
        )


class TestBlackwellAutoWSConstraints(TestCase):
    def test_two_ctas_allows_all_pipeline_depths(self):
        kwargs = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "EPILOGUE_SUBTILE": 2,
            "DATA_PARTITION_FACTOR": 1,
            "TWO_CTAS": True,
            "USE_META_WS": True,
        }
        with (
            config.patch({"triton.enable_template_tma_store": True}),
            unittest.mock.patch(
                "torch._inductor.heuristics.template.triton.has_two_ctas",
                return_value=True,
            ),
        ):
            for num_stages in range(2, 9):
                kwargs["num_stages"] = num_stages
                self.assertTrue(
                    CUDABlackwellPersistentTMATemplateConfigHeuristic._autows_constraints_ok(
                        kwargs, element_size=2
                    )
                )

    def test_two_ctas_swizzle_is_dtype_aware(self):
        kwargs = {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "EPILOGUE_SUBTILE": 1,
            "DATA_PARTITION_FACTOR": 1,
            "TWO_CTAS": True,
            "USE_META_WS": True,
        }
        with (
            config.patch({"triton.enable_template_tma_store": True}),
            unittest.mock.patch(
                "torch._inductor.heuristics.template.triton.has_two_ctas",
                return_value=True,
            ),
        ):
            self.assertFalse(
                CUDABlackwellPersistentTMATemplateConfigHeuristic._autows_constraints_ok(
                    kwargs, element_size=2
                )
            )
            self.assertTrue(
                CUDABlackwellPersistentTMATemplateConfigHeuristic._autows_constraints_ok(
                    kwargs, element_size=4
                )
            )

            kwargs["BLOCK_N"] = 32
            self.assertFalse(
                CUDABlackwellPersistentTMATemplateConfigHeuristic._autows_constraints_ok(
                    kwargs, element_size=4
                )
            )

    def test_two_ctas_requires_tma_store_for_metaws_template(self):
        kwargs = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "EPILOGUE_SUBTILE": 1,
            "DATA_PARTITION_FACTOR": 1,
            "TWO_CTAS": True,
            "USE_META_WS": True,
        }
        with (
            config.patch({"triton.enable_template_tma_store": False}),
            unittest.mock.patch(
                "torch._inductor.heuristics.template.triton.has_two_ctas",
                return_value=True,
            ),
        ):
            self.assertFalse(
                CUDABlackwellPersistentTMATemplateConfigHeuristic._autows_constraints_ok(
                    kwargs, element_size=2
                )
            )

    def test_two_ctas_odd_num_sms_covers_every_tile(self):
        with (
            unittest.mock.patch("torch.xpu.is_available", return_value=False),
            unittest.mock.patch(
                "torch._inductor.utils.get_max_num_sms", return_value=149
            ),
            unittest.mock.patch.object(
                torch._C, "_get_sm_carveout_experimental", return_value=None
            ),
        ):
            self.assertEqual(get_num_sms(), 149)
            num_sms = get_num_sms(two_ctas=True)
        self.assertEqual(num_sms, 148)

        block_m, block_n = 128, 128
        m, n = 17 * block_m, 20 * block_n
        meta = {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "NUM_SMS": num_sms,
            "TWO_CTAS": True,
        }
        grid_size = blackwell_persistent_mm_grid(m, n, meta)[0]
        grid_m = ((m + block_m - 1) // block_m + 1) // 2 * 2
        num_tiles = grid_m * ((n + block_n - 1) // block_n)
        visited = {
            tile for pid in range(grid_size) for tile in range(pid, num_tiles, num_sms)
        }
        self.assertEqual(visited, set(range(num_tiles)))


@unittest.skipIf(torch.version.hip is not None, "CUDA-specific template heuristics")
@instantiate_parametrized_tests
class TestBlackwellAutoWSConfigs(TestCase):
    """autoWS config selection for the Blackwell persistent-TMA template."""

    @parametrize(
        "op_name,heuristic_cls",
        (
            ("mm", CUDABlackwellPersistentTMATemplateConfigHeuristic),
            ("addmm", CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic),
            ("scaled_mm", CUDAScaledBlackwellTMATemplateConfigHeuristic),
        ),
    )
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    @parametrize("initial_autows", (False, True))
    def test_autows_configs_follow_current_mode(
        self, op_name, heuristic_cls, search_space, initial_autows
    ):
        with (
            mock.patch.dict(_HEURISTIC_CACHE, clear=True),
            mock.patch.dict(BaseHeuristicSingleton._instances, clear=True),
            mock.patch("torch._inductor.heuristics.template.triton.USE_META_WS", True),
        ):
            first_heuristic = None
            for enabled in (initial_autows, not initial_autows, initial_autows):
                with config.patch(
                    {
                        "triton.enable_template_autows": enabled,
                        "max_autotune_gemm_search_space": search_space,
                    }
                ):
                    heuristic = heuristic_cls()
                    if first_heuristic is None:
                        first_heuristic = heuristic
                    self.assertIs(heuristic, first_heuristic)
                    self.assertIs(
                        get_template_heuristic(
                            blackwell_ws_persistent_tma_mm_template.uid,
                            "cuda",
                            op_name,
                        ),
                        heuristic,
                    )
                    configs = heuristic._get_config_generator().keywords["configs"]
                    self.assertEqual(
                        {getattr(cfg, "use_meta_ws", False) for cfg in configs},
                        {enabled},
                    )
                    if enabled:
                        expected = (
                            heuristic._generate_autows_exhaustive_configs()
                            if search_space == "EXHAUSTIVE"
                            else heuristic._generate_autows_configs()
                        )
                        self.assertEqual(configs, expected)
                    else:
                        self.assertIs(
                            configs,
                            heuristic.exhaustive_configs
                            if search_space == "EXHAUSTIVE"
                            else heuristic.mm_configs,
                        )
                        if op_name == "addmm":
                            self.assertEqual(
                                heuristic.mm_configs,
                                heuristic.blackwell_persistent_mm_configs
                                + heuristic.blackwell_persistent_addmm_configs,
                            )

    @parametrize("global_meta_ws", (False, True))
    def test_global_meta_ws_disables_flatten(self, global_meta_ws):
        # Triton's Meta WS knob rewrites every WS kernel, so configs with
        # use_meta_ws=False must not be flattened either.
        mat1, mat2 = mock.Mock(), mock.Mock()
        mat2.get_dtype.return_value = torch.bfloat16
        kernel_inputs = MMKernelInputs([mat1, mat2], mat1_idx=0, mat2_idx=1)
        base = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "num_stages": 3,
            "num_warps": 4,
            "WARP_SPECIALIZE": True,
            "FLATTEN": True,
            "USE_META_WS": False,
        }
        with (
            mock.patch.dict(BaseHeuristicSingleton._instances, clear=True),
            mock.patch(
                "torch._inductor.heuristics.template.triton.USE_META_WS",
                global_meta_ws,
            ),
            mock.patch.object(
                TMATemplateConfigMixin,
                "_get_template_configs_impl",
                return_value=iter([base]),
            ),
        ):
            heuristic = CUDABlackwellPersistentTMATemplateConfigHeuristic()
            configs = list(heuristic._get_template_configs_impl(kernel_inputs, "mm"))
        self.assertEqual(len(configs), 1)
        self.assertTrue(configs[0]["WARP_SPECIALIZE"])
        self.assertEqual(configs[0]["FLATTEN"], not global_meta_ws)


@instantiate_parametrized_tests
class TestBlackwellBMMTemplate(TestCase):
    def _compile_blackwell_bmm(
        self,
        fn,
        *args,
        epilogue_subtile: int | None = None,
        host_side_tma: bool = False,
        expect_choice: bool = True,
        dynamic: bool = False,
    ):
        """Compile ``fn`` with ``blackwell_bmm`` lowered to one template choice."""

        class TestBlackwellBMMHeuristic(CUDABlackwellBMMTemplateConfigHeuristic):
            def _get_template_configs_impl(self, kernel_inputs, op_name):
                for template_config in super()._get_template_configs_impl(
                    kernel_inputs, op_name
                ):
                    if template_config["BLOCK_M"] == 128:
                        if epilogue_subtile is not None:
                            template_config["EPILOGUE_SUBTILE"] = epilogue_subtile
                        yield template_config
                        return

        def lowering(a_node, b_node):
            choices = V.choices.get_template_configs(
                MMKernelInputs([a_node, b_node]),
                [blackwell_ws_persistent_tma_bmm_template],
                "bmm",
            )
            if not expect_choice:
                self.assertEqual(choices, [])
                raise NoValidChoicesError("Blackwell BMM template rejected the input")
            return choices[0].output_node()

        with (
            override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[
                    (blackwell_ws_persistent_tma_bmm_template.uid, "bmm")
                ],
                override_heuristic_class=TestBlackwellBMMHeuristic,
            ),
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm.default: lowering},
            ),
            config.patch(
                compile_threads=1,
                **{"triton.enable_host_side_tma": host_side_tma},
            ),
        ):
            return run_and_get_code(
                torch.compile(fn, fullgraph=True, dynamic=dynamic), *args
            )

    @staticmethod
    def _blackwell_bmm_operand(bsz, rows, cols, *, broadcast, transpose):
        shape = (cols, rows) if transpose else (rows, cols)
        if not broadcast:
            shape = (bsz, *shape)
        t = torch.randn(shape, device=GPU_TYPE, dtype=torch.bfloat16)
        if transpose:
            t = t.transpose(-1, -2)
        return t.expand(bsz, rows, cols)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("epilogue_subtile", (1, 2, 4))
    @parametrize("transpose_a", (False, True))
    @parametrize("transpose_b", (False, True))
    @parametrize("broadcast", ("none", "a", "b", "ab"))
    @parametrize("host_side_tma", (False, True))
    def test_blackwell_bmm_template(
        self,
        epilogue_subtile: int,
        transpose_a: bool,
        transpose_b: bool,
        broadcast: str,
        host_side_tma: bool,
    ):
        if epilogue_subtile != 1 and (broadcast != "none" or host_side_tma):
            self.skipTest("epilogue subtiling does not depend on the operand loads")
        if host_side_tma and broadcast == "ab":
            self.skipTest("broadcast operands always use device-side descriptors")
        if (broadcast == "a" and transpose_b) or (broadcast == "b" and transpose_a):
            self.skipTest("the other operand's layouts are covered without broadcast")
        # 160 tiles exceed the SM count, so workers move between batches; each
        # batch has two M tiles, and K = 264 leaves a partial K tile.
        bsz, m, k, n = 80, 256, 264, 128
        a = self._blackwell_bmm_operand(
            bsz,
            m,
            k,
            broadcast="a" in broadcast,
            transpose=transpose_a,
        )
        b = self._blackwell_bmm_operand(
            bsz,
            k,
            n,
            broadcast="b" in broadcast,
            transpose=transpose_b,
        )

        def compile_bmm(host_side_tma):
            torch._dynamo.reset()
            return self._compile_blackwell_bmm(
                blackwell_bmm,
                a,
                b,
                epilogue_subtile=epilogue_subtile,
                host_side_tma=host_side_tma,
            )

        actual, code = compile_bmm(host_side_tma)
        torch.testing.assert_close(actual, torch.bmm(a, b), atol=1e-2, rtol=1e-2)
        if host_side_tma:
            # Host-built descriptors must match the device-side path exactly.
            self.assertEqual(actual, compile_bmm(False)[0], atol=0, rtol=0)
        code = code[0]
        # Rank-3 operands go through tma_descriptor(); broadcast operands keep
        # a device-side rank-2 descriptor.
        for name in ("A", "B"):
            desc = f"{name.lower()}_desc"
            if name.lower() in broadcast:
                self.assertIn(f"{desc} = triton.language.make_tensor_descriptor", code)
            elif host_side_tma:
                self.assertIn(f"{desc} = {name}\n", code)
                self.assertIn("host_tma_descriptor_args", code)
            else:
                self.assertRegex(
                    code,
                    rf"{desc} = tl\.make_tensor_descriptor\(base={name}, "
                    r"shape=\[\d+, \d+, \d+\], strides=\[\d+, \d+, 1\], "
                    r"block_shape=\[1, \d+, \d+\]\)",
                )
        # One global persistent queue over batches and tiles.
        FileCheck().check_not("tl.program_id(1)").run(code)
        FileCheck().check("num_tiles = BATCH * num_tiles_per_batch").check(
            "for tile_id in tl.range"
        ).run(code)
        self.assertNotIn("two_ctas=True", code)
        self.assertIn(f"EPILOGUE_SUBTILE : tl.constexpr = {epilogue_subtile}", code)
        self.assertIn("ALLOW_TF32 : tl.constexpr = False", code)
        if meta_ws_enabled():
            self.assertIn("DATA_PARTITION_FACTOR : tl.constexpr = 1", code)
            self.assertIn("SEPARATE_EPILOGUE_STORE : tl.constexpr = True", code)
        else:
            self.assertNotIn("data_partition_factor", code)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("broadcast_a", (False, True))
    def test_blackwell_bmm_template_data_partition_needs_broadcast_a(
        self, broadcast_a: bool
    ):
        # Data partitioning mis-slices rank-3 A loads, so a DPF > 1 config is
        # only offered when A is batch-broadcast (rank-2 descriptor).
        a = torch.randn(256, 256, device=GPU_TYPE, dtype=torch.bfloat16)
        a = a.expand(4, -1, -1) if broadcast_a else a.repeat(4, 1, 1)
        b = torch.randn(4, 256, 128, device=GPU_TYPE, dtype=torch.bfloat16)
        configs = (BlackwellBMMConfig(128, 128, 64, 4, 8, data_partition_factor=2),)
        with mock.patch.object(
            CUDABlackwellBMMTemplateConfigHeuristic, "bmm_configs", configs
        ):
            if not broadcast_a:
                with self.assertRaisesRegex(
                    BackendCompilerFailed, "rejected the input"
                ):
                    self._compile_blackwell_bmm(
                        blackwell_bmm, a, b, expect_choice=False
                    )
                return
            actual, _ = self._compile_blackwell_bmm(blackwell_bmm, a, b)
        torch.testing.assert_close(actual, torch.bmm(a, b), atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("fp32_precision", ("ieee", "tf32"))
    @parametrize("n", (128, 512))
    def test_blackwell_bmm_template_allow_tf32(self, fp32_precision: str, n: int):
        # fp32 follows the shared TF32 gate, including min(N, K) >= 512.
        expect_tf32 = fp32_precision == "tf32" and n >= 512
        if meta_ws_enabled() and not expect_tf32:
            self.skipTest("Meta autoWS fails to warp-specialize IEEE fp32 dots")
        a = torch.randint(-2, 3, (2, 256, 512), device=GPU_TYPE).float()
        b = torch.randint(-2, 3, (2, 512, n), device=GPU_TYPE).float()
        old_precision = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = fp32_precision
        try:
            actual, code = self._compile_blackwell_bmm(blackwell_bmm, a, b)
        finally:
            torch.backends.cuda.matmul.fp32_precision = old_precision
        # Small integers are exact in both TF32 and IEEE fp32.
        self.assertEqual(actual, torch.bmm(a, b), atol=0, rtol=0)
        self.assertIn(f"ALLOW_TF32 : tl.constexpr = {expect_tf32}", code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("aliased", (False, True))
    def test_blackwell_bmm_template_host_side_tma_fallback(self, aliased: bool):
        # A storage offset or an operand passed twice cannot be described by a
        # host descriptor, so both operands keep device descriptors.
        x = torch.randint(-2, 3, (5, 256, 256), device=GPU_TYPE).bfloat16()
        y = torch.randint(-2, 3, (4, 256, 256), device=GPU_TYPE).bfloat16()
        if aliased:
            fn, args, expected = lambda x: blackwell_bmm(x, x), (x,), torch.bmm(x, x)
        else:
            fn, args, expected = (
                lambda x, y: blackwell_bmm(x[1:], y),
                (x, y),
                torch.bmm(x[1:], y),
            )
        actual, code = self._compile_blackwell_bmm(fn, *args, host_side_tma=True)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertIn("HOST_SIDE_TMA : tl.constexpr = False", code[0])
        self.assertIn("a_desc = tl.make_tensor_descriptor(base=A", code[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize(
        "bsz,m,k,n,dynamic,batch_inner",
        (
            (2, 256, 256, 257, False, False),  # misaligned leading stride
            (2, 256, 0, 128, False, False),  # zero K
            (2, 256, 256, 128, True, False),  # dynamic shapes
            # Batch-innermost A: every stride is 16-byte aligned, but no matrix
            # dim is contiguous.
            (8, 256, 256, 128, False, True),
        ),
    )
    def test_blackwell_bmm_template_rejects(
        self,
        bsz: int,
        m: int,
        k: int,
        n: int,
        dynamic: bool,
        batch_inner: bool,
    ):
        a = torch.randn(bsz, m, k, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(bsz, k, n, device=GPU_TYPE, dtype=torch.bfloat16)
        if batch_inner:
            a = torch.randn(m, k, bsz, device=GPU_TYPE, dtype=torch.bfloat16)
            a = a.permute(2, 0, 1)
        with self.assertRaisesRegex(BackendCompilerFailed, "rejected the input"):
            self._compile_blackwell_bmm(
                blackwell_bmm, a, b, expect_choice=False, dynamic=dynamic
            )

    def _assert_blackwell_bmm_2cta_rejected(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        flatten_output: bool,
        heuristic_class: type = _Test2CTABlackwellBMMHeuristic,
    ) -> None:
        class FlatBMMKernelInputs(MMKernelInputs):
            def output_layout(self, flexible=True):
                return FixedLayout(
                    self.device(),
                    self.out_dtype(),
                    [a.shape[0] * a.shape[1], b.shape[2]],
                    [b.shape[2], 1],
                )

        def lowering(a_node, b_node):
            kernel_inputs_cls = (
                FlatBMMKernelInputs if flatten_output else MMKernelInputs
            )
            choices = V.choices.get_template_configs(
                kernel_inputs_cls([a_node, b_node]),
                [blackwell_ws_persistent_tma_bmm_template],
                "bmm",
            )
            self.assertEqual(choices, [])
            raise NoValidChoicesError("Blackwell BMM template was correctly rejected")

        custom_op = (
            torch.ops.inductor_test.blackwell_bmm_flat.default
            if flatten_output
            else torch.ops.inductor_test.blackwell_bmm.default
        )
        fn = blackwell_bmm_flat if flatten_output else blackwell_bmm
        with (
            self.assertRaisesRegex(BackendCompilerFailed, "correctly rejected"),
            override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[
                    (blackwell_ws_persistent_tma_bmm_template.uid, "bmm")
                ],
                override_heuristic_class=heuristic_class,
            ),
            mock.patch.dict(lowerings, {custom_op: lowering}),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            torch.compile(fn, fullgraph=True)(a, b)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipUnless(meta_ws_enabled(), "2CTA Blackwell BMM requires MetaWS")
    @parametrize("m", (256, 512))
    def test_blackwell_bmm_template_2cta_flat_output(self, m: int) -> None:
        bsz, k, n = 2, 8193, 128
        a_storage = torch.randn(bsz, k, m, device=GPU_TYPE, dtype=torch.bfloat16)
        a = a_storage.transpose(1, 2)
        b = torch.randn(bsz, k, n, device=GPU_TYPE, dtype=torch.bfloat16)

        class FlatBMMKernelInputs(MMKernelInputs):
            def output_layout(self, flexible=True):
                return FixedLayout(
                    self.device(), self.out_dtype(), [bsz * m, n], [n, 1]
                )

        def lowering(a_node, b_node):
            choices = V.choices.get_template_configs(
                FlatBMMKernelInputs([a_node, b_node]),
                [blackwell_ws_persistent_tma_bmm_template],
                "bmm",
            )
            return choices[0].output_node()

        with (
            override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[
                    (blackwell_ws_persistent_tma_bmm_template.uid, "bmm")
                ],
                override_heuristic_class=_Test2CTABlackwellBMMHeuristic,
            ),
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm_flat.default: lowering},
            ),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            actual, codes = run_and_get_code(
                torch.compile(blackwell_bmm_flat, fullgraph=True), a, b
            )

        torch.testing.assert_close(
            actual.view(bsz, m, n), torch.bmm(a, b), atol=1e-2, rtol=1e-2
        )
        self.assertIn("TWO_CTAS : tl.constexpr = True", codes[0])
        self.assertIn("ctas_per_cga=(2, 1, 1)", codes[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipUnless(meta_ws_enabled(), "2CTA Blackwell BMM requires MetaWS")
    def test_blackwell_bmm_template_2cta_odd_sm_count(self) -> None:
        # An odd SM count (e.g. from an SM carveout) must not leave tiles
        # unvisited: the kernel strides by NUM_SMS, so 2CTA rounds it to even.
        bsz, m, k, n = 8, 1024, 256, 1024
        # Small integers keep the fp32 accumulation exact.
        a = torch.randint(-1, 2, (bsz, m, k), device=GPU_TYPE).bfloat16()
        b = torch.randint(-1, 2, (bsz, k, n), device=GPU_TYPE).bfloat16()

        class FlatBMMKernelInputs(MMKernelInputs):
            def output_layout(self, flexible=True):
                return FixedLayout(
                    self.device(), self.out_dtype(), [bsz * m, n], [n, 1]
                )

        def lowering(a_node, b_node):
            choices = V.choices.get_template_configs(
                FlatBMMKernelInputs([a_node, b_node]),
                [blackwell_ws_persistent_tma_bmm_template],
                "bmm",
            )
            return choices[0].output_node()

        with (
            override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[
                    (blackwell_ws_persistent_tma_bmm_template.uid, "bmm")
                ],
                override_heuristic_class=_Test2CTABlackwellBMMHeuristic,
            ),
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm_flat.default: lowering},
            ),
            mock.patch.object(torch._inductor.utils, "get_max_num_sms", lambda: 147),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            actual, codes = run_and_get_code(
                torch.compile(blackwell_bmm_flat, fullgraph=True), a, b
            )

        self.assertEqual(actual.view(bsz, m, n), torch.bmm(a, b), atol=0, rtol=0)
        self.assertIn("TWO_CTAS : tl.constexpr = True", codes[0])
        self.assertIn("NUM_SMS : tl.constexpr = 146", codes[0])
        self.assertIn("make_tensor_descriptor(out_ptr0", codes[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("tail_case", ((127, 1), (200, 2)))
    def test_blackwell_bmm_template_1cta_rank3_tma_output(
        self, tail_case: tuple[int, int]
    ) -> None:
        m, epilogue_subtile = tail_case
        # M is not a multiple of BLOCK_M, so a rank-2 output descriptor would let
        # each tile's M tail spill into the next batch. Small integers keep the
        # result exact in bf16.
        bsz, k, n = 3, 256, 136
        a = torch.randint(-1, 2, (bsz, m, k), device=GPU_TYPE).to(torch.bfloat16)
        b = torch.randint(-1, 2, (bsz, k, n), device=GPU_TYPE).to(torch.bfloat16)
        expected = torch.bmm(a.float(), b.float()).bfloat16()

        class Rank3Output1CTABlackwellBMMHeuristic(
            CUDABlackwellBMMTemplateConfigHeuristic
        ):
            bmm_configs = (
                BlackwellBMMConfig(
                    128, 128, 64, 4, 8, epilogue_subtile=epilogue_subtile
                ),
            )

        def lowering(a_node, b_node):
            choices = V.choices.get_template_configs(
                MMKernelInputs([a_node, b_node]),
                [blackwell_ws_persistent_tma_bmm_template],
                "bmm",
            )
            self.assertEqual(len(choices), 1)
            return choices[0].output_node()

        with (
            override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[
                    (blackwell_ws_persistent_tma_bmm_template.uid, "bmm")
                ],
                override_heuristic_class=Rank3Output1CTABlackwellBMMHeuristic,
            ),
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm.default: lowering},
            ),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            compiled = torch.compile(blackwell_bmm, fullgraph=True)
            actual, codes = run_and_get_code(compiled, a, b)
            # Poison the freed output so the next call's allocation, which
            # reuses it, would expose any element the kernel fails to write.
            poisoned_ptr = actual.data_ptr()
            actual.fill_(float("nan"))
            del actual
            actual = compiled(a, b)

        self.assertEqual(actual.data_ptr(), poisoned_ptr)
        self.assertEqual(actual, expected, atol=0, rtol=0)
        self.assertIn("TWO_CTAS : tl.constexpr = False", codes[0])
        self.assertIn("RANK3_TMA_OUTPUT : tl.constexpr = True", codes[0])
        self.assertIn("make_tensor_descriptor(out_ptr0", codes[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipUnless(meta_ws_enabled(), "2CTA Blackwell BMM requires MetaWS")
    def test_blackwell_bmm_template_2cta_rejects_m_tail(self) -> None:
        bsz, m, k, n = 2, 384, 8193, 128
        a = torch.randn(bsz, m, k, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(bsz, k, n, device=GPU_TYPE, dtype=torch.bfloat16)
        self._assert_blackwell_bmm_2cta_rejected(
            a,
            b,
            flatten_output=True,
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @unittest.skipUnless(meta_ws_enabled(), "2CTA Blackwell BMM requires MetaWS")
    @parametrize("tail_case", ((127, 1), (129, 2), (384, 4)))
    def test_blackwell_bmm_template_2cta_rank3_output(
        self, tail_case: tuple[int, int]
    ) -> None:
        m, epilogue_subtile = tail_case
        bsz, k, n = 3, 4104, 136
        a = torch.randn(bsz, m, k, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(bsz, k, n, device=GPU_TYPE, dtype=torch.bfloat16)

        class Rank3Output2CTABlackwellBMMHeuristic(
            CUDABlackwellBMMTemplateConfigHeuristic
        ):
            bmm_configs = (
                BlackwellBMMConfig(
                    128,
                    128,
                    64,
                    4,
                    8,
                    epilogue_subtile=epilogue_subtile,
                    two_ctas=True,
                ),
            )

        def lowering(a_node, b_node):
            choices = V.choices.get_template_configs(
                MMKernelInputs([a_node, b_node]),
                [blackwell_ws_persistent_tma_bmm_template],
                "bmm",
            )
            self.assertEqual(len(choices), 1)
            return choices[0].output_node()

        with (
            override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[
                    (blackwell_ws_persistent_tma_bmm_template.uid, "bmm")
                ],
                override_heuristic_class=Rank3Output2CTABlackwellBMMHeuristic,
            ),
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm.default: lowering},
            ),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            compiled = torch.compile(blackwell_bmm, fullgraph=True)
            actual, codes = run_and_get_code(compiled, a, b)
            for _ in range(20):
                actual = compiled(a, b)

        torch.testing.assert_close(actual, torch.bmm(a, b), atol=1e-2, rtol=1e-2)
        self.assertIn("TWO_CTAS : tl.constexpr = True", codes[0])
        self.assertIn("RANK3_TMA_OUTPUT : tl.constexpr = True", codes[0])
        self.assertIn("ctas_per_cga=(2, 1, 1)", codes[0])
        self.assertIn("make_tensor_descriptor(out_ptr0", codes[0])

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_bmm_template_flat_output_rejects_m_tail(self) -> None:
        bsz, m, k, n = 3, 200, 256, 136
        a = torch.randn(bsz, m, k, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(bsz, k, n, device=GPU_TYPE, dtype=torch.bfloat16)
        self._assert_blackwell_bmm_2cta_rejected(
            a,
            b,
            flatten_output=True,
            heuristic_class=_Test1CTABlackwellBMMHeuristic,
        )

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_blackwell_bmm_template_rank3_output_m_tail(self) -> None:
        bsz, m, k, n = 3, 200, 256, 136
        # Small integers keep the fp32 accumulation exact.
        a = torch.randint(-1, 2, (bsz, m, k), device=GPU_TYPE).bfloat16()
        b = torch.randint(-1, 2, (bsz, k, n), device=GPU_TYPE).bfloat16()
        expected = torch.bmm(a.float(), b.float()).bfloat16()

        def lowering(a_node, b_node):
            choices = V.choices.get_template_configs(
                MMKernelInputs([a_node, b_node]),
                [blackwell_ws_persistent_tma_bmm_template],
                "bmm",
            )
            return choices[0].output_node()

        with (
            override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[
                    (blackwell_ws_persistent_tma_bmm_template.uid, "bmm")
                ],
                override_heuristic_class=_Test1CTABlackwellBMMHeuristic,
            ),
            mock.patch.dict(
                lowerings,
                {torch.ops.inductor_test.blackwell_bmm.default: lowering},
            ),
            config.patch(compile_threads=1),
        ):
            fn = torch.compile(blackwell_bmm, fullgraph=True)
            actual, codes = run_and_get_code(fn, a, b)
            # Poison the freed block so that rows the kernel skips stay NaN.
            poisoned_ptr = actual.data_ptr()
            actual.fill_(float("nan"))
            del actual
            actual = fn(a, b)

        self.assertEqual(actual.data_ptr(), poisoned_ptr)
        self.assertIn("blackwell_bmm", codes[0])
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @staticmethod
    def _offered_bmm_choices():
        from torch._inductor.kernel import bmm as bmm_module

        offered = []
        select = bmm_module.autotune_select_algorithm

        def spy(name, choices, *args, **kwargs):
            offered.extend(choices)
            return select(name, choices, *args, **kwargs)

        return offered, mock.patch.object(bmm_module, "autotune_select_algorithm", spy)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    @parametrize("op", ("bmm", "baddbmm"))
    def test_tuned_bmm_offers_blackwell_bmm_template(self, op: str):
        # M and N tails exercise the masked bias loads and the rank-3 store.
        bsz, m, k, n = 6, 200, 264, 136
        # Small integers and power-of-two scalars keep the result exact.
        alpha, beta = 2.0, 0.5
        a = torch.randint(-1, 2, (bsz, m, k), device=GPU_TYPE).bfloat16()
        b = torch.randint(-1, 2, (bsz, k, n), device=GPU_TYPE).bfloat16()
        bias = torch.randint(-4, 5, (bsz, m, n), device=GPU_TYPE).bfloat16()

        if op == "bmm":

            def fn(a, b, bias):
                return torch.bmm(a, b)

        else:

            def fn(a, b, bias):
                return torch.baddbmm(bias, a, b, alpha=alpha, beta=beta)

        expected = fn(a.double(), b.double(), bias.double()).bfloat16()
        counters.clear()
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "ATEN,TRITON",
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "blackwell_bmm",
                "compile_threads": 1,
            }
        ):
            actual, codes = run_and_get_code(torch.compile(fn), a, b, bias)

        # Lowered by tuned_{op}, i.e. post_grad did not unfuse baddbmm to bmm + add.
        self.assertTrue(
            any(key.startswith(f"aten.{op}_") for key in counters["aten_mm_info"])
        )
        # The persistent queue over batches and tiles is unique to the template.
        self.assertIn("num_tiles = BATCH * num_tiles_per_batch", codes[0])
        self.assertIn(f"USE_META_WS : tl.constexpr = {meta_ws_enabled()}", codes[0])
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_tuned_baddbmm_blackwell_bmm_beta_zero_ignores_nonfinite_bias(self):
        alpha = 2
        bsz, m, k, n = 4, 256, 256, 128
        a = torch.randint(-1, 2, (bsz, m, k), device=GPU_TYPE).bfloat16()
        b = torch.randint(-1, 2, (bsz, k, n), device=GPU_TYPE).bfloat16()
        bias = torch.full((bsz, m, n), float("nan"), device=GPU_TYPE).bfloat16()
        bias[1], bias[2] = float("inf"), float("-inf")

        def fn(bias, a, b):
            return torch.baddbmm(bias, a, b, alpha=alpha, beta=0)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "ATEN,TRITON",
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "blackwell_bmm",
                "compile_threads": 1,
            }
        ):
            actual, codes = run_and_get_code(torch.compile(fn), bias, a, b)

        self.assertIn("num_tiles = BATCH * num_tiles_per_batch", codes[0])
        expected = fn(bias.double(), a.double(), b.double()).bfloat16()
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_tuned_baddbmm_blackwell_bmm_bias_aliases_operand(self):
        # One kernel arg cannot carry both A's host descriptor and the bias pointer.
        bsz, m, k, n = 4, 256, 256, 128
        a = torch.randint(-1, 2, (bsz, m, k), device=GPU_TYPE).bfloat16()
        b = torch.randint(-1, 2, (bsz, k, n), device=GPU_TYPE).bfloat16()

        def fn(a, b):
            return torch.baddbmm(a[:, :, :n], a, b)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "ATEN,TRITON",
                "triton.enable_persistent_tma_matmul": True,
                "triton.enable_template_tma_store": True,
                "triton.enable_host_side_tma": True,
                "test_configs.autotune_choice_name_regex": "blackwell_bmm",
                "compile_threads": 1,
            }
        ):
            actual, codes = run_and_get_code(torch.compile(fn), a, b)

        self.assertIn("num_tiles = BATCH * num_tiles_per_batch", codes[0])
        self.assertIn("HOST_SIDE_TMA : tl.constexpr = False", codes[0])
        expected = fn(a.float(), b.float()).bfloat16()
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @unittest.skipIf(
        not has_datacenter_blackwell_tma_device(),
        "Need Blackwell with device-side TMA support in Triton",
    )
    def test_tuned_bmm_blackwell_bmm_template_rejects_symbolic_stride(self):
        a = torch.randn(4, 256, 384, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(4, 128, 128, device=GPU_TYPE, dtype=torch.bfloat16)
        # Static sizes, symbolic batch stride: a slice of a dynamic last dim.
        torch._dynamo.mark_dynamic(a, 2)

        def fn(a, b):
            return torch.bmm(a[:, :, :128], b)

        offered, spy = self._offered_bmm_choices()
        with (
            spy,
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "ATEN,TRITON",
                    "triton.enable_persistent_tma_matmul": True,
                    "compile_threads": 1,
                }
            ),
        ):
            actual = torch.compile(fn)(a, b)

        self.assertTrue(offered)
        self.assertFalse(any("blackwell_bmm" in c.name for c in offered))
        torch.testing.assert_close(actual, fn(a, b), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
