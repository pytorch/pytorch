# Owner(s): ["module: inductor"]
import unittest
from unittest import mock

import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor import config
from torch._inductor.autows_utils import meta_ws_enabled
from torch._inductor.heuristics.registry import (
    _HEURISTIC_CACHE,
    get_template_heuristic,
    override_template_heuristics,
)
from torch._inductor.heuristics.template.bmm import (
    CUDABlackwellBMMTemplateConfigHeuristic,
)
from torch._inductor.heuristics.template.triton import (
    BaseHeuristicSingleton,
    BlackwellGPUGemmConfig,
    CUDABlackwellAddmmPersistentTMATemplateConfigHeuristic,
    CUDABlackwellPersistentTMATemplateConfigHeuristic,
    CUDAScaledBlackwellTMATemplateConfigHeuristic,
    TMATemplateConfigMixin,
)
from torch._inductor.ir import FixedLayout
from torch._inductor.kernel.bmm import (
    blackwell_ws_persistent_tma_bmm_template,
    BlackwellBMMConfig,
)
from torch._inductor.kernel.mm import blackwell_ws_persistent_tma_mm_template
from torch._inductor.kernel.mm_common import blackwell_persistent_mm_grid
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.lowering import lowerings
from torch._inductor.select_algorithm import NoValidChoicesError
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import get_num_sms, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._triton import has_datacenter_blackwell_tma_device


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
        if host_side_tma:
            fc.check("host_tma_descriptor_args")
            if not tma_store:
                fc.check_not("tl.make_tensor_descriptor")
        else:
            fc.check("tl.make_tensor_descriptor")
        fc.check(write_api).run(code[0])
        if host_side_tma and tma_store:
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

    # NOTE: the current Inductor template verifies that the scaling mode is either per-tensor or per-row
    # TODO: support additional scaling modes for Blackwell
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
        if host_side_tma:
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
        if host_side_tma:
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
        if host_side_tma:
            fc.check("host_tma_descriptor_args")
            if not tma_store:
                fc.check_not(make_desc_api)
        else:
            fc.check(make_desc_api)
        fc.check(read_api).check(write_api).run(code[0])
        if host_side_tma and tma_store:
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


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
