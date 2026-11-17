# Owner(s): ["module: inductor"]
import math
import re
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch._inductor import config
from torch._inductor.autows_utils import meta_ws_enabled
from torch._inductor.ir import Buffer, ExternKernel, FixedLayout, FlexibleLayout
from torch._inductor.kernel.decompose_k import (
    BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS,
    DecomposeKSubgraphTemplate,
    get_blackwell_decompose_k_splits,
    lower_blackwell_decompose_k_partial,
)
from torch._inductor.kernel.mm_common import mm_args
from torch._inductor.lowering import lowerings, register_lowering
from torch._inductor.select_algorithm import autotune_select_algorithm, TritonTemplate
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.utils._triton import has_datacenter_blackwell_tma_device


def decomposeK(a, b, kPartitions):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    B = k // kPartitions
    a_reshaped = torch.permute(a.reshape(m, B, kPartitions), (1, 0, 2))
    b_reshaped = b.reshape(B, kPartitions, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    result_fp32 = result.to(torch.float32)
    reduced_buf = torch.sum(result_fp32, 0)
    return reduced_buf.to(a.dtype)


BLACKWELL_K_SPLIT = 8


@torch.library.custom_op("inductor_test::split_k_mm", mutates_args={})
def split_k_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


@split_k_mm.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


# Exposes the internal partial-BMM lowering directly, the only path where its
# operands can still have flexible layouts.
@torch.library.custom_op(
    "inductor_test::blackwell_decompose_k_partial", mutates_args={}
)
def blackwell_decompose_k_partial(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    m_pad = math.ceil(m / 128) * 128
    k_part = math.ceil(math.ceil(k / BLACKWELL_K_SPLIT) / 128) * 128
    out = a.new_zeros((BLACKWELL_K_SPLIT, m_pad, b.shape[1]), dtype=torch.float32)
    for split in range(BLACKWELL_K_SPLIT):
        begin, end = split * k_part, min((split + 1) * k_part, k)
        out[split, :m] = torch.mm(
            a[:, begin:end], b[begin:end], out_dtype=torch.float32
        )
    return out.view(-1, b.shape[1])


@blackwell_decompose_k_partial.register_fake
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m_pad = math.ceil(a.shape[0] / 128) * 128
    return a.new_empty((BLACKWELL_K_SPLIT * m_pad, b.shape[1]), dtype=torch.float32)


def _exact_inputs(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    # Small integers keep every partial sum exact, so results are bitwise.
    return torch.randint(-2, 3, shape, device=GPU_TYPE).to(dtype)


@instantiate_parametrized_tests
class TestSubgraphChoice(TestCase):
    def setUp(self):
        super().setUp()

    def _create_buffer(self, name, shape, dtype):
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape),
        )

    @decorateIf(
        unittest.skip("MetaWS and partial configs only apply to the Triton backend"),
        lambda params: params["backend"] == "aten"
        and (params["meta_ws"] or params["config_index"]),
    )
    @parametrize("backend", ("aten", "triton"))
    @parametrize("meta_ws", (False, True))
    @parametrize("config_index", (0, 1))
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    def test_subgraph_decompose_k(self, backend, meta_ws, config_index, dtype):
        if backend == "aten" and (meta_ws or config_index):
            self.skipTest("MetaWS and partial configs only apply to the Triton backend")
        if backend == "triton":
            if not has_datacenter_blackwell_tma_device():
                self.skipTest("the Triton partial BMM needs datacenter Blackwell")
            if meta_ws != meta_ws_enabled():
                self.skipTest(
                    f"covered when run with TRITON_USE_META_WS={int(meta_ws)}"
                )
        # Config 1 is a 2CTA config, which only runs as 2CTA under MetaWS.
        two_ctas = backend == "triton" and meta_ws and config_index == 1

        def lowering(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)
            choices = []
            DecomposeKSubgraphTemplate().maybe_append_choice(
                choices,
                k_split=BLACKWELL_K_SPLIT,
                input_nodes=(mat1, mat2),
                layout=layout,
                bmm_backend=backend,
                bmm_config_index=config_index if backend == "triton" else -1,
            )
            self.assertEqual(len(choices), 1)
            node, _ = autotune_select_algorithm(
                "test_subgraph_decompose_k", choices, [a, b], layout
            )
            return node

        # Two shapes with the same split in one graph. The first has an M tail and
        # an odd M-tile count (padded to a full pair under 2CTA), and for Triton
        # its last K partition is short for both partial configs.
        a1, b1 = (
            _exact_inputs(300, 8200, dtype=dtype),
            _exact_inputs(8200, 136, dtype=dtype),
        )
        a2, b2 = (
            _exact_inputs(256, 8200, dtype=dtype),
            _exact_inputs(8200, 128, dtype=dtype),
        )

        def fn(a1, b1, a2, b2):
            mm = torch.ops.inductor_test.split_k_mm
            return mm(a1, b1), mm(a2, b2)

        with (
            mock.patch.dict(
                lowerings, {torch.ops.inductor_test.split_k_mm.default: lowering}
            ),
            config.patch(
                compile_threads=1,
                **{
                    "triton.enable_template_tma_store": True,
                    "triton.enable_persistent_tma_matmul": True,
                },
            ),
        ):
            outputs, codes = run_and_get_code(torch.compile(fn), a1, b1, a2, b2)

        for out, a, b in zip(outputs, (a1, a2), (b1, b2)):
            self.assertEqual(out, (a.float() @ b.float()).to(dtype), atol=0, rtol=0)
        source = "\n".join(codes)
        # Subgraph functions are shared by name, so each shape needs its own.
        geometry_hashes = re.findall(
            r"decompose_k_mm_\d+_split_[a-z]+_([0-9a-f]{12})", source
        )
        self.assertEqual(len(set(geometry_hashes)), 2)
        if backend == "triton":
            self.assertIn(f"TWO_CTAS : tl.constexpr = {two_ctas}", source)
            self.assertIn("b_desc = triton.language.make_tensor_descriptor", source)
            self.assertIn("b = b_desc.load", source)
            self.assertNotIn("prologue_descriptor", source)

    @unittest.skipUnless(
        has_datacenter_blackwell_tma_device(),
        "the Triton partial BMM needs datacenter Blackwell",
    )
    def test_decompose_k_partial_freezes_producer_layout(self):
        # The partial bakes operand strides into constexprs; a producer's
        # flexible layout must be frozen first or the strides go stale.
        partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[0]

        def lowering(a, b):
            a, b = (ExternKernel.realize_input(t) for t in (a, b))
            m, k = map(int, a.get_size())
            m_pad = math.ceil(m / partial_config.block_m) * partial_config.block_m
            k_part = math.ceil(math.ceil(k / BLACKWELL_K_SPLIT) / 128) * 128
            return lower_blackwell_decompose_k_partial(
                a, b, BLACKWELL_K_SPLIT, 0, m_pad, k_part
            )

        # Column-major A keeps its leading stride TMA-aligned with K = 8193.
        a = _exact_inputs(8193, 256, dtype=torch.bfloat16).T
        y = _exact_inputs(128, 8193, dtype=torch.bfloat16)

        def fn(x, y):
            partial = torch.ops.inductor_test.blackwell_decompose_k_partial(
                x, y.t() * 2
            )
            return partial.view(BLACKWELL_K_SPLIT, 256, 128).sum(0).to(x.dtype)

        with (
            mock.patch.dict(
                lowerings,
                {
                    torch.ops.inductor_test.blackwell_decompose_k_partial.default: lowering
                },
            ),
            config.patch(
                compile_threads=1, **{"triton.enable_template_tma_store": True}
            ),
        ):
            actual = torch.compile(fn)(a, y)
        expected = (a.float() @ (y.t() * 2).float()).to(a.dtype)
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @unittest.skipUnless(
        has_datacenter_blackwell_tma_device(),
        "the Triton partial BMM needs datacenter Blackwell",
    )
    @parametrize(
        "outer,nested,expect_aten,expect_triton",
        (
            ("ATEN", "ATEN,TRITON", False, False),  # outer ATEN-only: no decompose-K
            ("TRITON", None, True, False),  # nested default is ATEN
            ("TRITON", "triton", False, True),  # case-insensitive, Triton only
            ("TRITON", "aten, TRITON", True, True),  # both plans enumerated
        ),
    )
    def test_decompose_k_bmm_backends(self, outer, nested, expect_aten, expect_triton):
        a = _exact_inputs(131072, 256, dtype=torch.bfloat16).T
        b = _exact_inputs(131072, 128, dtype=torch.bfloat16)
        patch = {
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": outer,
            "compile_threads": 1,
            "triton.enable_template_tma_store": True,
            "triton.enable_persistent_tma_matmul": True,
            "triton.num_decompose_k_splits": 4,
            "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
        }
        if nested is not None:
            patch["triton.decompose_k_bmm_backends"] = nested
        with config.patch(patch):
            actual, codes = run_and_get_code(
                torch.compile(lambda x, y: x @ y, fullgraph=True), a, b
            )
        self.assertEqual(actual, (a.float() @ b.float()).to(a.dtype), atol=0, rtol=0)
        source = "\n".join(codes)
        self.assertEqual("_split_aten" in source, expect_aten)
        self.assertEqual("_split_triton_" in source, expect_triton)

    def test_backend_specific_split_candidates(self):
        one_cta = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[0]
        two_cta_wide = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[2]

        # Geometry only: the per-split K bound is covered separately below.
        def splits(m, n, k, partial_config):
            return get_blackwell_decompose_k_splits(
                m, n, k, 148, partial_config, max_k_per_split=0
            )

        production_ks = (
            10_879_109,
            10_954_007,
            11_091_857,
            11_047_127,
            11_029_264,
        )
        for k in production_ks:
            self.assertEqual(splits(256, 128, k, one_cta), [74, 148])

        self.assertEqual(splits(128, 128, 11_047_127, one_cta), [148, 296])
        self.assertEqual(splits(128, 256, 969_147, one_cta), [74, 149])
        self.assertEqual(splits(256, 424, 973_138, two_cta_wide), [37, 74])
        self.assertEqual(splits(512, 424, 11_047_127, two_cta_wide), [19, 37])

        # Exact and uneven K use the same bounded geometry-based candidates,
        # while shallow K and a lone M tile reject the 2CTA schedule.
        self.assertEqual(splits(256, 424, 973_248, two_cta_wide), [37, 74])
        self.assertEqual(splits(256, 128, 8_193, one_cta), [])
        self.assertEqual(splits(128, 424, 11_047_127, two_cta_wide), [])

    def test_blackwell_split_candidates_bound_k_per_split(self):
        # The tile-wave generator never calls get_k_splits, so it applies
        # decompose_k_max_k_per_split itself. g027-like shape: unbounded, split
        # 10 gives 27456-long fp32 partial accumulations.
        one_cta = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[0]
        two_cta_wide = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[2]

        def k_parts(k, ks, partial_config):
            block_k = partial_config.block_k
            return [math.ceil(math.ceil(k / s) / block_k) * block_k for s in ks]

        with config.patch({"triton.decompose_k_max_k_per_split": 16384}):
            ks = get_blackwell_decompose_k_splits(512, 896, 274_456, 148, two_cta_wide)
            self.assertEqual(ks, [17, 19])
            self.assertTrue(all(p <= 16384 for p in k_parts(274_456, ks, two_cta_wide)))
            # A K so long that every wave-sized split exceeds the bound starts
            # the search at the smallest split that fits it.
            ks = get_blackwell_decompose_k_splits(256, 128, 11_047_127, 148, one_cta)
            self.assertEqual(ks, [675])
            self.assertEqual(k_parts(11_047_127, ks, one_cta), [16384])
        with config.patch({"triton.decompose_k_max_k_per_split": 0}):
            self.assertEqual(
                get_blackwell_decompose_k_splits(512, 896, 274_456, 148, two_cta_wide),
                [10, 19],
            )

    @unittest.skipUnless(
        has_datacenter_blackwell_tma_device(),
        "the Triton partial BMM needs datacenter Blackwell",
    )
    def test_1cta_fused_producer_uses_descriptor_loads(self):
        m, k, n = 128, 262_145, 256
        a = torch.randn(k, m, device=GPU_TYPE, dtype=torch.bfloat16).T
        x = torch.randn(k, n, device=GPU_TYPE, dtype=torch.float32)
        gate = torch.randn(k, n, device=GPU_TYPE, dtype=torch.float32)

        def fn(a, x, gate):
            b = (x * torch.sigmoid(gate)).to(torch.bfloat16)
            return a @ b

        with (
            mock.patch.object(TritonTemplate, "test_cache", True),
            config.patch(
                max_autotune_gemm=True,
                max_autotune_gemm_backends="TRITON",
                benchmark_epilogue_fusion=True,
                compile_threads=1,
                assume_aligned_inputs=True,
                **{
                    "triton.enable_template_tma_store": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.num_decompose_k_splits": 2,
                    "triton.decompose_k_bmm_backends": "TRITON",
                    "test_configs.autotune_choice_name_regex": "_triton_",
                },
            ),
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True), a, x, gate
            )

        torch.testing.assert_close(actual, fn(a, x, gate), atol=16.0, rtol=1e-1)
        source = codes[-1]
        self.assertIn("tl.sigmoid", source)
        self.assertIn("prologue_descriptor0 = tl.make_tensor_descriptor", source)
        self.assertIn("prologue_descriptor1 = tl.make_tensor_descriptor", source)
        self.assertGreaterEqual(
            source.count("shape=[262145, 256], strides=[256, 1]"), 2
        )
        self.assertIn("prologue_descriptor0.load([offs_k, offs_bn])", source)
        self.assertIn("prologue_descriptor1.load([offs_k, offs_bn])", source)
        self.assertIn("warp_specialize=WARP_SPECIALIZE", source)
        self.assertIn("BLOCK_K : tl.constexpr = 64", source)
        self.assertIn("TWO_CTAS : tl.constexpr = False", source)
        self.assertIn("num_stages=4", source)
        self.assertNotIn("arg_B", source)
        self.assertIn("triton_red_fused", source)

    @unittest.skipUnless(
        has_datacenter_blackwell_tma_device(),
        "the Triton partial BMM needs datacenter Blackwell",
    )
    def test_fused_producer_aten_plan_fallback(self):
        m, k, n = 128, 65_536, 256
        a = torch.randn(k, m, device=GPU_TYPE, dtype=torch.bfloat16).T
        x = torch.randn(k, n, device=GPU_TYPE, dtype=torch.float32)
        gate = torch.randn(k, n, device=GPU_TYPE, dtype=torch.float32)

        def fn(a, x, gate):
            return a @ (x * torch.sigmoid(gate)).to(torch.bfloat16)

        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="ATEN",
            benchmark_epilogue_fusion=True,
            compile_threads=1,
            **{
                "triton.num_decompose_k_splits": 2,
                "triton.decompose_k_bmm_backends": "ATEN",
            },
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True), a, x, gate
            )

        torch.testing.assert_close(actual, fn(a, x, gate), atol=16.0, rtol=1e-1)
        source = "\n".join(codes)
        self.assertNotIn("prologue_descriptor", source)
        self.assertIn("extern_kernels.mm", source)

    @unittest.skipUnless(
        has_datacenter_blackwell_tma_device(),
        "the Triton partial BMM needs datacenter Blackwell",
    )
    def test_1cta_fused_producer_preserves_shared_input(self):
        m, k, n = 128, 262_145, 256
        a = torch.randn(k, m, device=GPU_TYPE, dtype=torch.bfloat16).T
        x = torch.randn(k, n, device=GPU_TYPE, dtype=torch.bfloat16)
        gate = torch.randn(k, n, device=GPU_TYPE, dtype=torch.float32)

        def fn(a, x, gate):
            sigmoid = torch.sigmoid(gate)
            return a @ (x.float() * sigmoid).to(torch.bfloat16), sigmoid

        with (
            mock.patch.object(TritonTemplate, "test_cache", True),
            config.patch(
                max_autotune_gemm=True,
                max_autotune_gemm_backends="TRITON",
                benchmark_epilogue_fusion=True,
                compile_threads=1,
                assume_aligned_inputs=True,
                **{
                    "triton.enable_template_tma_store": True,
                    "triton.enable_persistent_tma_matmul": True,
                    "triton.num_decompose_k_splits": 2,
                    "triton.decompose_k_bmm_backends": "TRITON",
                    "test_configs.autotune_choice_name_regex": "_triton_",
                },
            ),
        ):
            actual, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True), a, x, gate
            )

        expected = fn(a, x, gate)
        torch.testing.assert_close(actual[0], expected[0], atol=16.0, rtol=1e-1)
        torch.testing.assert_close(actual[1], expected[1])
        source = codes[-1]
        self.assertIn("tl.sigmoid", source)
        self.assertIn("prologue_descriptor0.load([offs_k, offs_bn])", source)
        self.assertIn("prologue_descriptor1.load([offs_k, offs_bn])", source)
        self.assertNotIn("arg_B", source)
        # The shared sigmoid is materialized; the BF16 MMA operand is not.
        self.assertEqual(source.count("empty_strided_cuda((262145, 256)"), 1)

    def test_subgraph_freeze_layout(self):
        from torch._inductor.kernel.decompose_k import DecomposeKSubgraphTemplate
        from torch._inductor.kernel.mm_common import mm_args

        M, N, K = (4, 128, 14240)
        a_in = torch.randn(
            (M, K), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            (K, N), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )

        @torch.library.custom_op("mylib::matmul_decompose_padding", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose_padding)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)
            mat1_layout = mat1.layout
            if not isinstance(mat1_layout, FlexibleLayout):
                raise AssertionError
            mat1_stride = mat1_layout.stride

            choices = []

            kPartitions = 2

            decompose_k_subgraph_template = DecomposeKSubgraphTemplate()

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            choice = choices[0]
            if not isinstance(mat1.layout, FixedLayout):
                raise AssertionError

            # Creating the subgraph choice should have frozen the layout
            # We ensure padding so the stride should differ
            if mat1.layout.stride == mat1_stride:
                raise AssertionError

            for example_stride, layout_stride in zip(
                choice.example_inputs[0].stride(), mat1.layout.stride
            ):
                # Example inputs should have same stride as current layout
                if example_stride != layout_stride:
                    raise AssertionError

            node, _ = autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )
            return node

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose_padding((mat1 + 1.0), mat2)

        with mock.patch("torch._inductor.ir.V.get_current_node") as get_node_mock:
            node_mock = MagicMock()
            node_mock.meta = {"dislike_padding": False}
            get_node_mock.return_value = node_mock

            compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

            compiled_func(a_in, b_in)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()
