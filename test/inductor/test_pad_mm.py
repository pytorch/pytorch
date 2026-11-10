# Owner(s): ["module: inductor"]
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters
from torch._inductor import utils as inductor_utils
from torch._inductor.fx_passes.pad_mm import (
    _bf16_large_k_needs_pad,
    _pad_device_key,
    _pad_pass_measured,
    can_pad,
    check_device,
    get_alignment_size,
    get_do_bench,
    get_pad_cache,
    get_padded_length,
    is_mm_compute_bound,
    should_pad,
    should_pad_bench_key,
    should_pad_mm_bf16,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, is_big_gpu, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


@unittest.skipIf(
    not HAS_GPU_AND_TRITON,
    "PadMMTest benches padded mm end to end and needs an accelerator with Triton",
)
class PadMMTest(TestCase):
    def setUp(self):
        super().setUp()
        if not is_big_gpu():
            return self.skipTest("Need a big GPU to run max_autotune=True")

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_mm_dyn_m(self):
        M = 40
        K1 = 581
        K2 = 49
        N = 30

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = rand_strided(
                    (K2, N), (1, K2), device=GPU_TYPE, dtype=torch.float32
                )

            def forward(self, a):
                a1 = torch.narrow(a, 1, 0, K2)
                return torch.mm(a1, self.w)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K1), (K1, 1), device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        torch._dynamo.mark_dynamic(a, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_cat_pad_mm_dyn_m(self):
        M1 = 128
        M2 = 40
        K1 = 129
        K2 = 111
        N = 100

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = rand_strided(
                    (K2, N), (1, K2), device=GPU_TYPE, dtype=torch.float32
                )

            def forward(self, a, b):
                c = torch.cat([a, b], dim=0)
                a1 = torch.narrow(c, 1, 0, K2)
                return torch.mm(a1, self.w)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M1, K1), (K1, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((M2, K1), (K1, 1), device=GPU_TYPE, dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        aligned_k = get_padded_length(K2, get_alignment_size(a)) + K2
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_mm_dyn_n(self):
        M = 20
        K = 81
        N = 30

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.mm(a, b)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K), (K, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_mm_dyn_k(self):
        M = 21
        K = 80
        N = 30

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.mm(a, b)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K), (K, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float32)
        # TODO: Getting the alignment right requires pattern matcher to
        # run on newly added nodes
        aligned_m = get_padded_length(M, get_alignment_size(a)) + M
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"M = {aligned_m}").run(code)
        self.assertEqual(res1, res2)

    def test_pad_mm_dyn_mnk(self):
        M = 20
        K = 81
        N = 30

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.mm(a, b)

        fn = Model().to(GPU_TYPE)
        a = rand_strided((M, K), (K, 1), device=GPU_TYPE, dtype=torch.float32)
        b = rand_strided((K, N), (1, K), device=GPU_TYPE, dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (_,) = run_and_get_code(compiled_fn, a, b)
        self.assertEqual(res1, res2)

    @inductor_config.patch(force_shape_pad=True)
    def test_zero_dim(self):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        self.assertEqual(torch.compile(addmm)(x, a, b), addmm(x, a, b))

    @fresh_cache()
    @inductor_config.patch(shape_padding=True)
    @unittest.skipIf(
        GPU_TYPE != "cuda",
        "CUDA eager ignores addmm input shape when beta=0",
    )
    def test_addmm_beta_zero_mismatched_bias_skips_padding(self):
        def addmm(bias, x, weight):
            return torch.addmm(bias, x, weight, beta=0.0, alpha=0.1)

        def bench(fn):
            fn()
            return 1.0

        bias = torch.zeros(8, device=GPU_TYPE)
        x = torch.randn(2, 8, device=GPU_TYPE)
        weight = torch.randn(8, 13, device=GPU_TYPE)

        with (
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                return_value=True,
            ),
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_do_bench",
                return_value=bench,
            ),
        ):
            actual = torch.compile(addmm, fullgraph=True)(bias, x, weight)
        self.assertEqual(actual, addmm(bias, x, weight))

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_bmm_dyn_b(self):
        B = 10
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(B, K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(a)) + K
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_bmm_dyn_k(self):
        B = 10
        M = 128
        K = 40
        N = 41

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(B, K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N
        torch._dynamo.mark_dynamic(a, 2)
        torch._dynamo.mark_dynamic(b, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"N = {aligned_n}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_bmm_dyn_bm(self):
        B = 10
        M = 128
        K = 40
        N = 41

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b):
                return torch.bmm(a, b)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(B, M, K, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(B, K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_n = get_padded_length(N, get_alignment_size(b)) + N
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b)
            FileCheck().check(f"N = {aligned_n}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_addmm_dyn_m(self):
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b, c):
                return torch.addmm(a, b, c)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(M, N, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float32)
        c = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float32)
        aligned_k = get_padded_length(K, get_alignment_size(b)) + K
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b, c)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b, c)
            FileCheck().check(f"K = {aligned_k}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(
        max_autotune=True, max_autotune_gemm_backends="TRITON", force_shape_pad=True
    )
    def test_pad_addmm_dyn_mn(self):
        M = 128
        K = 33
        N = 40

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b, c):
                return torch.addmm(a, b, c)

        fn = Model().to(GPU_TYPE)
        a = torch.randn(M, N, device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float32)
        c = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float32)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(c, 1)
        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            res1 = fn(a, b, c)
            compiled_fn = torch.compile(fn)
            res2, (code,) = run_and_get_code(compiled_fn, a, b, c)
            # no padding
            FileCheck().check(f"K = {K}").run(code)
        self.assertEqual(res1, res2)

    @inductor_config.patch(force_shape_pad=True)
    def test_pad_single_cat(self):
        @torch.compile()
        def foo(x, y):
            return x @ y

        inps = [torch.rand([5, 5], device=GPU_TYPE) for _ in range(2)]
        out = foo(*inps)
        self.assertEqual(out, inps[0] @ inps[1])

    @inductor_config.patch(force_shape_pad=True)
    @fresh_cache()
    def test_pad_addmm_2d_bias(self):
        @torch.compile()
        def foo(input, x, y):
            return torch.ops.aten.addmm(input, x, y)

        for a in [1, 4]:
            for b in [1, 6]:
                inps = (
                    torch.rand([a, b], device=GPU_TYPE),
                    torch.rand([4, 5], device=GPU_TYPE),
                    torch.rand([5, 6], device=GPU_TYPE),
                )
                out = foo(*inps)
                out_eager = torch.ops.aten.addmm(*inps)
                self.assertEqual(out, out_eager)

        for a in [1, 6]:
            inps = (
                torch.rand([a], device=GPU_TYPE),
                torch.rand([4, 5], device=GPU_TYPE),
                torch.rand([5, 6], device=GPU_TYPE),
            )
            out = foo(*inps)
            out_eager = torch.ops.aten.addmm(*inps)
            self.assertEqual(out, out_eager)

    @inductor_config.patch(force_shape_pad=True)
    def test_pad_batch(self):
        m = 6
        n = 9
        k = 11
        batch_size = 3
        mat1 = torch.ones((batch_size, m, k), device=GPU_TYPE, dtype=torch.float16)
        mat2 = torch.ones((batch_size, k, n), device=GPU_TYPE, dtype=torch.float16)
        expected_alignment = get_alignment_size(mat1)

        if expected_alignment != 8:
            raise AssertionError("Alignment for float16 should be 8")
        if not can_pad(mat1, mat2, torch.ops.aten.bmm):
            raise AssertionError("This should pass the common padding criteria")

        @torch.compile()
        def bmm(mat1, mat2):
            return torch.bmm(mat1, mat2)

        res2, (code,) = run_and_get_code(bmm, mat1, mat2)
        bmm_expected_result = torch.bmm(mat1, mat2)
        # in call code, expect to see a single pad per input, and then we should see padded allocation for output
        FileCheck().check("del async_compile").check_count(
            ".run(", 2, exactly=True
        ).check(f"empty_strided_{GPU_TYPE}((3, 8, 16)").run(code)

        if not torch.allclose(res2, bmm_expected_result):
            raise AssertionError("BMM results are not identical")

    @fresh_cache()
    def test_exclude_padding(self):
        def bench(fn):
            fn()
            return 1.0

        @torch.compile()
        def mm(a, b):
            return a @ b

        # Size must require padding to 4 elements; the compute-bound heuristic is
        # patched below so the cache-key assertions do not depend on GPU model.
        size = [61, 61]
        with (
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.is_mm_compute_bound",
                return_value=True,
            ),
            unittest.mock.patch(
                "torch._inductor.fx_passes.pad_mm.get_do_bench",
                return_value=bench,
            ),
        ):
            mm(torch.rand(size, device=GPU_TYPE), torch.rand(size, device=GPU_TYPE))
            local_cache = get_pad_cache().get_local_cache()
            self.assertEqual(len(local_cache), 2)
            FileCheck().check_count("exclude_pad:False", 2, exactly=True).run(
                repr(local_cache)
            )

            @torch.compile()
            def mm(a, b):
                return (a + 1) @ b

            mm(torch.rand(size, device=GPU_TYPE), torch.rand(size, device=GPU_TYPE))
            local_cache = get_pad_cache().get_local_cache()
            # reuse original base timing
            self.assertEqual(len(local_cache), 3)

            FileCheck().check_count("exclude_pad:False", 3, exactly=True).run(
                repr(local_cache)
            )
            FileCheck().check_count("exclude_pad:True", 1, exactly=True).run(
                repr(local_cache)
            )

    @fresh_cache()
    @inductor_config.patch(max_pointwise_cat_inputs=2)
    def test_exclude_cat_padding(self):
        @torch.compile()
        def mm(inps, b):
            return torch.cat(inps) @ b

        inp = torch.rand([2046, 2046], device=GPU_TYPE)
        inp2 = torch.rand([2046, 2046], device=GPU_TYPE)

        inps = inp.chunk(3)
        mm(inps, inp2)
        FileCheck().check_count("exclude_pad:False", 2, exactly=True).run(
            repr(get_pad_cache().get_local_cache())
        )

        inps = inp.chunk(2)
        mm(inps, inp2)
        FileCheck().check_count("exclude_pad:False", 3, exactly=True).run(
            repr(get_pad_cache().get_local_cache())
        )

    @unittest.skipIf(
        not HAS_GPU_AND_TRITON or not _bf16_large_k_needs_pad(torch.device(GPU_TYPE)),
        "No bf16 large-K pad regression on this device",
    )
    @fresh_cache()
    @inductor_config.patch(
        post_grad_fusion_options={"pad_aten_mm_pass": {"k_threshold_to_pad": 8388608}}
    )
    def test_pad_mm_bf16(self):
        m = 2
        n = 13
        k = 15691904
        mat1 = torch.ones((m, k), device=GPU_TYPE, dtype=torch.bfloat16)
        mat2 = torch.ones((k, n), device=GPU_TYPE, dtype=torch.bfloat16)
        expected_alignment = get_alignment_size(mat1)

        if expected_alignment != 8:
            raise AssertionError("Alignment for bfloat16 should be 8")
        if not can_pad(mat1, mat2, torch.ops.aten.mm):
            raise AssertionError("This should pass the common padding criteria")
        if not should_pad_mm_bf16(mat1.dtype, m, n, k, mat1.device):
            raise AssertionError(
                "This should pass the should_pad_mm_bf16 padding criteria"
            )

        @torch.compile()
        def mm(mat1, mat2):
            return torch.mm(mat1, mat2)

        res2, (code,) = run_and_get_code(mm, mat1, mat2)
        mm_expected_result = torch.mm(mat1, mat2)
        # in call code, expect to see a single pad per input, and then we should see padded allocation for output
        FileCheck().check("del async_compile").check_count(
            ".run(", 2, exactly=True
        ).check(f"empty_strided_{GPU_TYPE}((8, 16)").run(code)

        if not torch.allclose(res2, mm_expected_result):
            raise AssertionError("MM results are not identical")

    @fresh_cache()
    @inductor_config.patch(
        {
            "triton.unique_kernel_names": "original_aten",
            "max_autotune_gemm_backends": "TRITON",
            # Force padding so it is applied regardless of the device-dependent
            # is_mm_compute_bound checks in should_pad_common.
            "force_shape_pad": True,
        }
    )
    def test_original_aten_preserved_pad_mm(self):
        def fn(x, y):
            return x @ y

        args = [
            torch.randn(2**4, 2**8 - 1, device=GPU_TYPE, dtype=torch.float16),
            torch.randn(2**8 - 1, 2**4, device=GPU_TYPE, dtype=torch.float16),
        ]

        counters.clear()

        with unittest.mock.patch(
            "torch._inductor.fx_passes.pad_mm._skip_do_bench_times", True
        ):
            opt_fn = torch.compile(fn, mode="max-autotune")
            ret, code = run_and_get_code(opt_fn, *args)
        # xref: https://github.com/pytorch/pytorch/pull/172780
        if not torch.version.hip:  # autotuning is not guaranteed to run on ROCm
            self.assertEqual(counters["inductor"]["pattern_matcher_count"], 1)

        code = [c for c in code if "decompose_k" not in c]
        # The mm kernel should use a template (because we set max_autotune_gemm_backends = TRITON).
        # Its name should contain `mm` because `mm` was the original aten op where the mm came from.
        FileCheck().check("def triton_tem_fused_mm").run(code[0])

    def test_no_autocast_in_pad_bmm_joint_graph_pass(self):
        # Track bmm dtypes before and after joint graph passes
        bmm_dtypes_pre = {}
        bmm_dtypes_post = {}

        def make_bmm_dtype_tracker(dtype_dict):
            def track_bmm_dtype(graph):
                for node in graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten.bmm.default
                    ):
                        # Store the output dtype
                        if hasattr(node.meta.get("val", None), "dtype"):
                            dtype_dict[str(node)] = node.meta["val"].dtype
                return graph

            return track_bmm_dtype

        class MaskedMHA(torch.nn.Module):
            def __init__(self, H_q, H_kv, D):
                super().__init__()
                self.H_kv = H_kv
                num_heads_total = H_q + 2 * H_kv
                self.qkv_proj_vid = torch.nn.Linear(H_q * D, num_heads_total * D)
                self.qkv_proj_txt = torch.nn.Linear(H_q * D, num_heads_total * D)
                self.out_proj = torch.nn.Linear(H_q * D, H_q * D)
                self.H_q = H_q
                self.D = D

            def forward(self, x_vid, x_txt, attn_mask):
                qkv_vid = self.qkv_proj_vid(x_vid)
                qkv_txt = self.qkv_proj_txt(x_txt)
                qkv_vid = qkv_vid.reshape((*qkv_vid.shape[:-1], -1, self.D))
                qkv_txt = qkv_txt.reshape((*qkv_txt.shape[:-1], -1, self.D))

                q_vid = qkv_vid[..., : self.H_q, :]
                k_vid = qkv_vid[..., self.H_q : self.H_q + self.H_kv, :]
                v_vid = qkv_vid[..., self.H_q + self.H_kv :, :]

                q_txt = qkv_txt[..., : self.H_q, :]
                k_txt = qkv_txt[..., self.H_q : self.H_q + self.H_kv, :]
                v_txt = qkv_txt[..., self.H_q + self.H_kv :, :]

                q = torch.cat([q_vid, q_txt], dim=-3)
                k = torch.cat([k_vid, k_txt], dim=-3)
                v = torch.cat([v_vid, v_txt], dim=-3)

                out = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(-2, -3),
                    k.transpose(-2, -3),
                    v.transpose(-2, -3),
                    attn_mask=attn_mask,
                    enable_gqa=True,
                )
                out = out.transpose(-2, -3)

                return out

        def test_masked_mha(B, H, S, D, device, dtype):
            S_vid = 300
            S_txt = S - S_vid
            x1 = torch.randn(B, S_vid, H * D, requires_grad=True, device=device)
            x2 = torch.randn(B, S_txt, H * D, requires_grad=True, device=device)
            attn_mask = torch.ones(B, 1, S, S, dtype=torch.bool, device=device)

            H_kv = H // 4
            mha = MaskedMHA(H, H_kv, D)
            mha = mha.to(device)

            with torch._inductor.config.patch(
                joint_custom_pre_pass=make_bmm_dtype_tracker(bmm_dtypes_pre),
                joint_custom_post_pass=make_bmm_dtype_tracker(bmm_dtypes_post),
            ):
                mha = torch.compile(mha, fullgraph=True, backend="inductor")
                with torch.autocast(
                    device_type=GPU_TYPE, dtype=dtype, cache_enabled=False
                ):
                    out_vid = mha(x1, x2, attn_mask)
                    target_vid = torch.randn_like(out_vid)

                    loss_vid = (out_vid - target_vid).mean()
                    loss = loss_vid
                loss.backward()

            torch.accelerator.synchronize()

            # Check if any bmm operations had dtype changes
            for node_name_pre, node_name_post in zip(
                bmm_dtypes_pre, bmm_dtypes_post, strict=True
            ):
                pre_dtype = bmm_dtypes_pre[node_name_pre]
                post_dtype = bmm_dtypes_post[node_name_post]
                # Assert no bmm output dtype changes
                self.assertEqual(pre_dtype, post_dtype)

            # Based on issue https://github.com/pytorch/pytorch/issues/159469,
            # if autocast was applied in pad_bmm causing bmm's output dtype to be changed from fp32 to bf16,
            # gradient will have NaNs in this test case.
            self.assertFalse(torch.any(x1.grad.isnan()).item())
            self.assertFalse(torch.any(x2.grad.isnan()).item())

        B, H, S, D = 2, 32, 549, 128
        device = GPU_TYPE
        dtype = torch.bfloat16
        torch.compiler.reset()
        torch.manual_seed(42)
        test_masked_mha(B, H, S, D, device, dtype)

    @inductor_config.patch(force_shape_pad=True, strict_output_strides=True)
    def test_pad_mm_output_strides_preserved(self):
        """Regression test: pad_mm creates views with padded strides.
        User-visible output strides must match eager execution."""

        def fn(x, y):
            return torch.mm(x, y)

        # N=2 gets padded to 4, so the mm output is (3, 4) sliced to (3, 2),
        # creating a view with stride (4, 1) instead of the expected (2, 1).
        x = torch.randn(3, 5, device=GPU_TYPE)
        y = torch.randn(5, 2, device=GPU_TYPE)
        expected = fn(x, y)
        compiled = torch.compile(fn)(x, y)
        self.assertEqual(compiled, expected)
        self.assertEqual(compiled.stride(), expected.stride())


@instantiate_parametrized_tests
class PadMMDeviceAgnosticTest(TestCase):
    """
    The pad decisions in pad_mm must key off the device of the tensors being
    compiled, not off which accelerators happen to be visible to the process.
    These cases run anywhere, including on CPU-only machines.
    """

    hw_classification = HardwareClassification.GENERIC

    # Mirrors test_pad_mm_bf16: bf16, K > M, K > N, N odd and K above the pad
    # threshold, so the only thing left to decide is the device.
    M, K, N = 2, 15691904, 13
    PAD_OPTIONS = {"pad_aten_mm_pass": {"k_threshold_to_pad": 8388608}}

    @staticmethod
    def _fake_device(device: str) -> torch.device:
        # A FakeTensorMode resolves a device string to a device object without
        # the hardware -- and without assuming which alias privateuse1 answers
        # to on this build.
        with torch._subclasses.FakeTensorMode():
            return torch.empty(4, 13, device=device).device

    def test_check_device_accepts_any_accelerator(self):
        # The criterion is a negation: any accelerator both operands agree on is
        # paddable, so a backend does not have to be named here to be reachable.
        # The interface hook, not check_device, then decides which of these
        # actually reach can_pad's gate (see test_can_pad_asks_the_operand_device).
        with torch._subclasses.FakeTensorMode():
            accelerators = [
                torch.empty(4, 4, device=d)
                for d in ("cuda", "xpu", "mps", "mtia", "privateuseone:0")
            ]

        for t in accelerators:
            self.assertTrue(check_device(t, t), f"{t.device} should be paddable")
        for i, a in enumerate(accelerators):
            for b in accelerators[i + 1 :]:
                self.assertFalse(check_device(a, b), f"{a.device} vs {b.device}")

        # Real tensors: cpu never pads, and neither does the meta device that
        # pattern-matcher inputs carry.
        self.assertFalse(check_device(torch.empty(4, 4), torch.empty(4, 4)))
        meta = torch.empty(4, 4, device="meta")
        self.assertFalse(check_device(meta, meta))
        self.assertFalse(check_device(accelerators[0], torch.empty(4, 4)))

    def test_pad_pass_measured_reads_the_interface_declaration(self):
        # The measured set is declared on each backend's device interface, not
        # in this file: cuda and xpu declare True, a registered backend that has
        # not measured inherits the base False, and a device type no backend
        # registers answers False instead of raising. Patching the hook on an
        # interface demonstrates the out-of-tree opt-in route (declare it on the
        # interface class after measuring padding on that hardware) without
        # touching the global registry.
        from torch._dynamo.device_interface import (
            CpuInterface,
            CudaInterface,
            MpsInterface,
            MtiaInterface,
            TpuInterface,
            XpuInterface,
        )

        self.assertTrue(CudaInterface.supports_pad_mm())
        self.assertTrue(XpuInterface.supports_pad_mm())
        for iface in (MtiaInterface, MpsInterface, TpuInterface, CpuInterface):
            self.assertFalse(iface.supports_pad_mm())
        for device, expected in (
            ("cuda:1", True),
            ("xpu", True),
            ("mtia", False),
            ("mps", False),
            ("vulkan:0", False),
        ):
            with self.subTest(device=device):
                self.assertEqual(_pad_pass_measured(torch.device(device)), expected)
        with unittest.mock.patch.object(
            MtiaInterface, "supports_pad_mm", staticmethod(lambda device=None: True)
        ):
            self.assertTrue(_pad_pass_measured(torch.device("mtia")))

    @inductor_config.patch(post_grad_fusion_options=PAD_OPTIONS)
    def test_should_pad_mm_bf16_asks_only_the_compiled_device(self):
        args = (torch.bfloat16, self.M, self.N, self.K)
        other = self._fake_device("privateuseone:0")
        with unittest.mock.patch(
            "torch.cuda.get_device_capability", return_value=(8, 0)
        ) as cuda_cap:
            self.assertTrue(should_pad_mm_bf16(*args, device=torch.device("cuda:1")))
            cuda_cap.assert_called_with(1)
            # xpu answers from the device table; any other accelerator takes the
            # default of "this regression was never seen here". Neither one
            # probes cuda's hardware.
            self.assertTrue(should_pad_mm_bf16(*args, device=torch.device("xpu")))
            self.assertFalse(should_pad_mm_bf16(*args, device=other))
            self.assertEqual(cuda_cap.call_count, 1)

    @inductor_config.patch(post_grad_fusion_options=PAD_OPTIONS)
    def test_should_pad_mm_bf16_survives_an_unanswerable_capability_probe(self):
        # The public helpers stay reachable directly, so a capability probe
        # that cannot answer must mean "no measured regression to force".
        # A build without CUDA and an invalid device index both raise
        # AssertionError, alongside the errors the neighbouring device
        # lookups (utils.get_device_tflops, _get_device_info_key) catch.
        args = (torch.bfloat16, self.M, self.N, self.K)
        probes = (
            (AssertionError("Torch not compiled with CUDA enabled"), "no_cuda_build"),
            (AssertionError("Invalid device id"), "invalid_index"),
            (IndexError("no such device"), "index_error"),
            (RuntimeError("driver"), "runtime_error"),
            (ValueError("bad value"), "value_error"),
        )
        for exc, name in probes:
            with self.subTest(probe=name):
                with unittest.mock.patch(
                    "torch.cuda.get_device_capability", side_effect=exc
                ):
                    self.assertFalse(
                        should_pad_mm_bf16(*args, device=torch.device("cuda:1"))
                    )

    @inductor_config.patch(post_grad_fusion_options=PAD_OPTIONS)
    def test_should_pad_mm_bf16_skips_pad_on_hopper(self):
        args = (torch.bfloat16, self.M, self.N, self.K)
        with unittest.mock.patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ):
            self.assertFalse(should_pad_mm_bf16(*args, device=torch.device("cuda:1")))

    def test_is_mm_compute_bound_bf16_large_k_is_device_specific(self):
        # These rates make the shape bandwidth bound, so a True can only come
        # from the device-specific bf16 large-K override.
        args = (self.M, self.K, self.N, torch.bfloat16)
        other = self._fake_device("privateuseone:0")
        with (
            unittest.mock.patch.object(
                inductor_utils, "get_device_tflops", return_value=100.0
            ) as tflops,
            unittest.mock.patch.object(
                inductor_utils, "get_device_dram_gbps", return_value=2000.0
            ) as gbps,
        ):
            with unittest.mock.patch(
                "torch.cuda.get_device_capability", return_value=(8, 0)
            ):
                self.assertTrue(is_mm_compute_bound(*args, device=torch.device("cuda")))
            with unittest.mock.patch(
                "torch.cuda.get_device_capability", return_value=(9, 0)
            ):
                self.assertFalse(
                    is_mm_compute_bound(*args, device=torch.device("cuda"))
                )
            self.assertTrue(is_mm_compute_bound(*args, device=torch.device("xpu")))
            self.assertFalse(is_mm_compute_bound(*args, device=other))
            # The roofline rates are looked up for the device being decided
            # about, so no decision reads another device's numbers.
            tflops.assert_called_with(torch.bfloat16, other)
            gbps.assert_called_with(other)

    def test_is_mm_compute_bound_without_roofline_info_defers_to_benchmark(self):
        # A device without roofline info reports tflops 0.0 or DRAM None; the
        # comparison is meaningless there, so the decision defers to the
        # measured benchmark (True) despite the bandwidth-bound mock rates.
        args = (self.M, self.K, self.N, torch.bfloat16)
        other = self._fake_device("privateuseone:0")
        for tflops_ret, dram_ret in ((0.0, 2000.0), (100.0, None)):
            with (
                unittest.mock.patch.object(
                    inductor_utils, "get_device_tflops", return_value=tflops_ret
                ) as tflops,
                unittest.mock.patch.object(
                    inductor_utils,
                    "get_device_dram_gbps",
                    return_value=dram_ret,
                ) as gbps,
            ):
                with self.subTest(tflops=tflops_ret, dram=dram_ret):
                    self.assertTrue(is_mm_compute_bound(*args, device=other))
                    # The rates come from the device-generic lookups, asked
                    # about the operand, not from a process-wide GPU query.
                    tflops.assert_called_with(torch.bfloat16, other)
                    gbps.assert_called_with(other)

    @parametrize("device", ["cuda", "cuda:1", "xpu", "mps", "privateuseone:0", "mtia"])
    def test_can_pad_asks_the_operand_device(self, device):
        # can_pad hands mat1.device -- the full device, index included -- to
        # the Triton gate, and only for a backend this pass has measured. A
        # measured device (cuda/xpu) is asked about by name; an unmeasured
        # accelerator (mps/privateuse1/mtia) is rejected by the hook before the
        # gate runs, so it is never padded. CPU and meta are not listed:
        # check_device answers them before the hook is consulted, and
        # test_check_device_* pins that.
        asked = []

        def fake_gate(gate_device):
            asked.append(gate_device)
            return True

        with torch._subclasses.FakeTensorMode():
            mat1 = torch.empty(4, 13, device=device)
            mat2 = torch.empty(13, 8, device=device)
        operand = mat1.device
        gate = "torch._inductor.fx_passes.pad_mm.has_triton_for_device"
        with (
            inductor_config.patch(shape_padding=True, deterministic=False),
            unittest.mock.patch(gate, side_effect=fake_gate),
        ):
            result = can_pad(mat1, mat2, torch.ops.aten.mm)

        measured = operand.type in ("cuda", "xpu")
        self.assertEqual(result, measured)
        self.assertEqual(asked, [operand] if measured else [])

    def test_deterministic_path_rejects_unmeasured_before_allocating(self):
        # With deterministic=True the autoheuristic gate is open by default, so
        # without a device check in can_pad an unmeasured operand would clear
        # can_pad and _should_pad would realize (allocate) it only to drop it at
        # the autoheuristic gate. The interface hook rejects it here instead;
        # cuda/xpu still proceed. The assertion is on can_pad alone: the
        # rejection happens before any operand is realized.
        gate = "torch._inductor.fx_passes.pad_mm.has_triton_for_device"
        for device, expected in (
            ("cuda", True),
            ("xpu", True),
            ("mtia", False),
            ("privateuseone:0", False),
        ):
            with self.subTest(device=device):
                with torch._subclasses.FakeTensorMode():
                    mat1 = torch.empty(4, 13, device=device)
                    mat2 = torch.empty(13, 8, device=device)
                with (
                    inductor_config.patch(shape_padding=True, deterministic=True),
                    unittest.mock.patch(gate, return_value=True),
                ):
                    self.assertEqual(can_pad(mat1, mat2, torch.ops.aten.mm), expected)

    def test_can_pad_reports_false_for_an_unusable_device_index(self):
        # A fake or exported graph naming a device this host doesn't have
        # (cuda:3 on a 2-GPU box) must make can_pad answer False, not raise
        # through the Triton gate the way an unvalidated per-index probe would.
        from torch._dynamo import device_interface as di
        from torch._dynamo.device_interface import DeviceInterface
        from torch.utils import _triton as triton_utils

        class _TwoGpu(DeviceInterface):
            # The hook declaration is what lets a cuda-class operand reach the
            # Triton gate; there the device_count pre-check answers cuda:3
            # before any probe runs, so no Triton probe is ever reached.

            @staticmethod
            def supports_pad_mm(device=None) -> bool:
                return True

            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 2

        with torch._subclasses.FakeTensorMode():
            mat1 = torch.empty(4, 13, device="cuda:3")
            mat2 = torch.empty(13, 8, device="cuda:3")
        with (
            inductor_config.patch(shape_padding=True, deterministic=False),
            unittest.mock.patch.object(
                triton_utils, "has_triton_package", return_value=True
            ),
            unittest.mock.patch.object(
                di, "get_interface_for_device", return_value=_TwoGpu
            ),
        ):
            self.assertFalse(can_pad(mat1, mat2, torch.ops.aten.mm))

    def test_bench_keys_are_scoped_by_hardware_not_ordinal(self):
        # The pad-decision key's device scope is hardware identity, not the
        # rank-local ordinal: cuda:0 and cuda:1 on the same GPU model share a
        # warm entry (DDP/FSDP ranks compile the same shape on cuda:0..N-1 and
        # share cache_dir()), a different hardware name or device type does not,
        # and a device with no resolvable name falls back to its type alone.
        with torch._subclasses.FakeTensorMode():
            a_cuda0 = torch.empty(4, 13, device="cuda:0")
            b_cuda0 = torch.empty(13, 8, device="cuda:0")
            a_cuda1 = torch.empty(4, 13, device="cuda:1")
            b_cuda1 = torch.empty(13, 8, device="cuda:1")
            a_other = torch.empty(4, 13, device="privateuseone:0")
            b_other = torch.empty(13, 8, device="privateuseone:0")
        op = torch.ops.aten.mm

        def key(mat1, mat2):
            # match is unused when is_base_time_key=True (fmt_pad short-circuits).
            return should_pad_bench_key(None, mat1, mat2, op, is_base_time_key=True)  # type: ignore[arg-type]

        def same_model(device):
            return "A100" if device.type == "cuda" else None

        with unittest.mock.patch.object(
            inductor_utils, "_get_device_info_key", side_effect=same_model
        ):
            self.assertEqual(key(a_cuda0, b_cuda0), key(a_cuda1, b_cuda1))
            self.assertNotEqual(key(a_cuda0, b_cuda0), key(a_other, b_other))

        def by_index(device):
            return {0: "A100", 1: "H100"}.get(device.index)

        with unittest.mock.patch.object(
            inductor_utils, "_get_device_info_key", side_effect=by_index
        ):
            self.assertNotEqual(key(a_cuda0, b_cuda0), key(a_cuda1, b_cuda1))

    def test_pad_device_key_scopes_by_type_when_the_worker_cannot_answer(self):
        # The opt-in route advertised on the device interface: a backend may
        # declare the hook and the Triton probes while inheriting the base
        # Worker, whose get_device_properties cannot answer. The pad cache
        # then scopes by device type alone; every bench-key caller inherits
        # that degradation through this same key helper.
        from torch._dynamo.device_interface import DeviceInterface

        class _NoWorker(DeviceInterface):
            pass

        device = torch.device("privateuseone:0")
        with unittest.mock.patch.object(
            inductor_utils, "get_interface_for_device", return_value=_NoWorker
        ):
            self.assertEqual(_pad_device_key(device), "privateuseone")

    def test_pad_device_key_degrades_when_the_name_lookup_raises(self):
        # The degradation is a contract of _pad_device_key itself, not a side
        # effect of the device lookup's current catch set: if the hardware-
        # name lookup propagates a NotImplementedError, the key falls back to
        # the device type instead of raising through every bench-key caller.
        device = torch.device("privateuseone:0")
        with unittest.mock.patch.object(
            inductor_utils, "_get_device_info_key", side_effect=NotImplementedError
        ):
            self.assertEqual(_pad_device_key(device), "privateuseone")

    def test_device_probes_receive_operand_index(self):
        # The operand's index must reach every hardware probe: a cuda:1 GPU
        # older than the cuda:0 GPU in the same host still gets padded, and the
        # roofline rates come from cuda:1.
        def fake_capability(device=None):
            index = device.index if isinstance(device, torch.device) else device
            return (8, 0) if index == 1 else (9, 0)

        cuda1 = torch.device("cuda:1")
        with (
            unittest.mock.patch(
                "torch.cuda.get_device_capability", side_effect=fake_capability
            ) as cuda_cap,
            unittest.mock.patch.object(
                inductor_utils, "get_device_tflops", return_value=100.0
            ) as tflops,
            unittest.mock.patch.object(
                inductor_utils, "get_device_dram_gbps", return_value=2000.0
            ) as gbps,
            inductor_config.patch(post_grad_fusion_options=self.PAD_OPTIONS),
        ):
            self.assertTrue(
                should_pad_mm_bf16(torch.bfloat16, self.M, self.N, self.K, cuda1)
            )
            self.assertFalse(
                should_pad_mm_bf16(
                    torch.bfloat16, self.M, self.N, self.K, torch.device("cuda:0")
                )
            )
            self.assertNotIn(unittest.mock.call(), cuda_cap.call_args_list)
            # fp16 skips the bf16 override and reaches the roofline lookups.
            self.assertFalse(
                is_mm_compute_bound(self.M, self.K, self.N, torch.float16, cuda1)
            )
            tflops.assert_called_with(torch.float16, cuda1)
            gbps.assert_called_with(cuda1)

    @fresh_cache()
    def test_should_pad_forwards_the_operand_device(self):
        # Drive the real entry point so the forwarded devices are pinned at the
        # call sites and not only at the callees: with a cuda:1 operand next to
        # a cuda:0 one, the index must survive down to every probe. fresh_cache()
        # starts from empty caches; the mocked hardware names then keep the two
        # operands on distinct bench keys, so even with the caching module
        # enabled the memoizer cannot replay one operand's answer onto the other.
        match = unittest.mock.MagicMock()
        match.output_node.return_value.meta = {}
        op = torch.ops.aten.mm

        def fake_capability(device=None):
            # get_device_capability resolves a missing index to the current
            # device; this host stands at cuda:0, a Hopper-class GPU that does
            # not pad, while cuda:1 is older and does.
            if isinstance(device, torch.device):
                device = device.index
            return (8, 0) if device == 1 else (9, 0)

        def fake_info_key(device):
            # Distinct hardware names per index: the bench key's device scope
            # must separate cuda:0 from cuda:1 when the caching module is on
            # and the memoizer is live, or the second call replays the first.
            return f"model-{device.index}"

        def operands(device: str):
            with torch._subclasses.FakeTensorMode():
                return (
                    torch.empty(self.M, self.K, device=device, dtype=torch.bfloat16),
                    torch.empty(self.K, self.N, device=device, dtype=torch.bfloat16),
                )

        pad_mod = "torch._inductor.fx_passes.pad_mm"
        triton_gate = f"{pad_mod}.has_triton_for_device"
        exclude_time = f"{pad_mod}.should_exclude_padding_time"
        do_bench_site = f"{pad_mod}.get_do_bench"
        bf16_site = f"{pad_mod}.should_pad_mm_bf16"
        bound_site = f"{pad_mod}.is_mm_compute_bound"
        capability = "torch.cuda.get_device_capability"
        tflops_site = "torch._inductor.utils.get_device_tflops"
        dram_site = "torch._inductor.utils.get_device_dram_gbps"
        info_key_site = "torch._inductor.utils._get_device_info_key"
        with (
            inductor_config.patch(
                shape_padding=True,
                force_shape_pad=False,
                deterministic=False,
                post_grad_fusion_options=self.PAD_OPTIONS,
            ),
            unittest.mock.patch(triton_gate, return_value=True),
            unittest.mock.patch(exclude_time, return_value=False),
            unittest.mock.patch(capability, side_effect=fake_capability),
            unittest.mock.patch(tflops_site, return_value=100.0),
            unittest.mock.patch(dram_site, return_value=2000.0),
            unittest.mock.patch(info_key_site, side_effect=fake_info_key),
            # The heuristics run for real; only the benchmark and the hardware
            # probes are stood in for, so the decision itself is on the table.
            unittest.mock.patch(do_bench_site, wraps=get_do_bench) as do_bench,
            unittest.mock.patch(bf16_site, wraps=should_pad_mm_bf16) as bf16,
            unittest.mock.patch(bound_site, wraps=is_mm_compute_bound) as bound,
        ):
            mat1, mat2 = operands("cuda:1")
            self.assertTrue(should_pad(match, mat1, mat2, op))
            do_bench.assert_called_with("cuda")
            bf16.assert_called_with(
                torch.bfloat16, self.M, self.N, self.K, torch.device("cuda:1")
            )

            do_bench.reset_mock()
            bf16.reset_mock()
            # Same shape on cuda:0: the bf16 override declines, so the decision
            # falls through to the roofline check, which is handed the operand's
            # device too, and the answer is "do not pad".
            mat1, mat2 = operands("cuda:0")
            self.assertFalse(should_pad(match, mat1, mat2, op))
            do_bench.assert_called_with("cuda")
            bf16.assert_called_with(
                torch.bfloat16, self.M, self.N, self.K, torch.device("cuda:0")
            )
            bound.assert_called_with(
                self.M, self.K, self.N, torch.bfloat16, torch.device("cuda:0")
            )

    def test_get_do_bench_forwards_device_type(self):
        # The device_type kwarg is what puts the benchmark's L2 flush buffer on
        # the device being decided about; get_do_bench takes the operand's
        # device *type* because the benchmarker collapses any torch.device to it.
        # The whole kwarg set is pinned: TritonBenchmarker strips kwargs absent
        # from do_bench, so a dropped or renamed one disappears silently instead
        # of failing.
        from torch._inductor.runtime.benchmarking import benchmarker

        with unittest.mock.patch.object(
            benchmarker, "benchmark_gpu", return_value=1.0
        ) as mock_bench:
            get_do_bench("xpu")(lambda: None)
        forwarded = mock_bench.call_args.kwargs
        self.assertEqual(forwarded, {"warmup": 5, "device_type": "xpu"})


if __name__ == "__main__":
    run_tests()
