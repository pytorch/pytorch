# Owner(s): ["module: inductor"]
import math
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.pad_mm import (
    can_pad,
    get_alignment_size,
    get_pad_cache,
    get_padded_length,
    should_pad_mm_bf16,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, is_big_gpu, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


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
        (not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (9, 0))
        and (not torch.xpu.is_available()),
        "No perf regression on H100+ with BF16",
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
        if not should_pad_mm_bf16(mat1.dtype, m, n, k):
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


class PadBaddbmmTest(TestCase):
    def setUp(self):
        super().setUp()
        self._inductor_test_stack.enter_context(
            inductor_config.patch(
                force_shape_pad=True,
                max_autotune=False,
                max_autotune_gemm=False,
                max_autotune_gemm_backends="ATEN",
            )
        )

    def _check_compiled(self, fn, args, dimensions):
        observed = []

        def record_graph(graph):
            for node in graph.nodes:
                if node.target is torch.ops.aten.baddbmm.default:
                    mat1, mat2 = (arg.meta["val"] for arg in node.args[1:3])
                    observed.append((mat1.shape[1], mat1.shape[2], mat2.shape[2]))

        expected = fn(*args)
        tangent = (
            torch.arange(expected.numel(), device=expected.device)
            .remainder(3)
            .add(1)
            .reshape(expected.shape)
            .to(expected.dtype)
            / 8
        )
        grad_args = tuple(arg for arg in args if arg.requires_grad)
        expected_grads = (
            torch.autograd.grad(expected, grad_args, tangent) if grad_args else ()
        )
        with inductor_config.patch(joint_custom_post_pass=record_graph):
            actual = torch.compile(fn, fullgraph=True)(*args)
        self.assertEqual(actual, expected)
        if grad_args:
            actual_grads = torch.autograd.grad(actual, grad_args, tangent)
            self.assertEqual(actual_grads, expected_grads)
        if dimensions is None:
            original_dims = (args[1].shape[1], args[1].shape[2], args[2].shape[2])
            for dims in observed:
                self.assertEqual(dims, original_dims)
        else:
            self.assertEqual(observed, [dimensions])
        return actual

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize(
        "shape,bias_shape,scalars",
        [
            ((2, 13, 11, 9), (2, 13, 9), {}),
            ((2, 13, 16, 16), (16,), {"alpha": -0.5, "beta": 2.0}),
            ((2, 16, 16, 9), (2, 1, 9), {"alpha": 0.0, "beta": 0.5}),
            ((2, 16, 11, 16), (), {"alpha": 0.5, "beta": 0.0}),
            ((2, 13, 11, 9), (13, 1), {"alpha": 2.0, "beta": -0.5}),
        ],
    )
    def test_padding_and_gradients(self, device, dtype, shape, bias_shape, scalars):
        batch, m, k, n = shape

        def values(size, scale):
            values = torch.arange(math.prod(size), device=device).remainder(5) - 2
            return values.to(dtype).reshape(size) * scale

        args = (
            values(bias_shape, 0.25),
            values((batch, m, k), 0.25),
            values((batch, k, n), 0.125),
        )
        for arg in args:
            arg.requires_grad_()
        if not bias_shape:
            self.assertFalse(
                can_pad(args[1][0], args[2][0], torch.ops.aten.addmm, args[0])
            )

        def fn(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2, **scalars)

        alignment = 4 if dtype is torch.float32 else 8
        dimensions = tuple(
            (dim + alignment - 1) // alignment * alignment for dim in shape[1:]
        )
        self._check_compiled(fn, args, dimensions)

    @parametrize("dynamic_dim", ["batch_and_m", "k"])
    def test_dynamic_shapes(self, device, dynamic_dim):
        def fn(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2, alpha=0.5, beta=2.0)

        observed = []

        def record_graph(graph):
            for node in graph.nodes:
                if node.target is torch.ops.aten.baddbmm.default:
                    mat1, mat2 = (arg.meta["val"] for arg in node.args[1:3])
                    observed.append((mat1.shape[1], mat1.shape[2], mat2.shape[2]))

        compiled = torch.compile(fn, fullgraph=True)
        for batch, m, k in [(2, 13, 11), (3, 17, 15)]:
            if dynamic_dim == "k":
                batch, m = 2, 13
            else:
                k = 11
            args = (
                torch.full((9,), 0.25, device=device),
                torch.full((batch, m, k), 0.25, device=device),
                torch.full((batch, k, 9), 0.125, device=device),
            )
            if dynamic_dim == "batch_and_m":
                torch._dynamo.mark_dynamic(args[1], 0)
                torch._dynamo.mark_dynamic(args[1], 1)
                torch._dynamo.mark_dynamic(args[2], 0)
            else:
                torch._dynamo.mark_dynamic(args[1], 2)
                torch._dynamo.mark_dynamic(args[2], 1)
            with inductor_config.patch(joint_custom_post_pass=record_graph):
                actual = compiled(*args)
            self.assertEqual(actual, fn(*args))
        self.assertTrue(observed)
        for m, k, n in observed:
            self.assertEqual(n, 12)
            if dynamic_dim == "batch_and_m":
                self.assertEqual(k, 12)
            else:
                self.assertEqual(m, 16)

    @parametrize(
        "layout,requires_grad",
        [
            ("offset_transpose", (False, True, True)),
            ("batch_broadcast", (True, True, False)),
        ],
    )
    def test_strided_inputs_and_partial_gradients(self, device, layout, requires_grad):
        batch, m, k, n = 2, 13, 11, 9

        def values(shape, scale):
            data = torch.arange(math.prod(shape), device=device).remainder(7) - 3
            return data.float().reshape(shape) * scale

        if layout == "offset_transpose":
            mat1 = values((batch, k + 2, m + 2), 0.25)[:, 1:-1, 1:-1].transpose(1, 2)
            mat2 = values((batch, n + 2, k + 2), 0.125)[:, 1:-1, 1:-1].transpose(1, 2)
        else:
            mat1 = values((1, m, k), 0.25).expand(batch, m, k)
            mat2 = values((batch, k, n), 0.125)
        bias = values((batch, 1, n), 0.25)
        args = (bias, mat1, mat2)
        originals = tuple(arg.clone() for arg in args)
        for arg, needs_grad in zip(args, requires_grad):
            arg.requires_grad_(needs_grad)

        def fn(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2, alpha=0.5, beta=-0.5)

        self._check_compiled(fn, args, (16, 12, 12))
        self.assertEqual(args, originals)

    @parametrize(
        "shape",
        [(0, 3, 5, 2), (2, 0, 5, 2), (2, 3, 0, 2), (2, 3, 5, 0), (2, 1, 1, 1)],
    )
    def test_empty_and_singleton_fallback(self, device, shape):
        batch, m, k, n = shape
        args = (
            torch.ones((n,), device=device),
            torch.ones((batch, m, k), device=device),
            torch.ones((batch, k, n), device=device),
        )
        self._check_compiled(torch.baddbmm, args, dimensions=None)

    @inductor_config.patch(strict_output_strides=True)
    @parametrize(
        "shape,requires_grad",
        [
            ((2, 3, 5, 2), False),
            ((2, 3, 5, 2), True),
            ((1, 3, 4, 4), True),
            ((1, 1, 5, 9), True),
        ],
    )
    def test_output_strides(self, device, shape, requires_grad):
        batch, m, k, n = shape
        args = (
            torch.ones((batch, m, n), device=device, requires_grad=requires_grad),
            torch.ones((batch, m, k), device=device, requires_grad=requires_grad),
            torch.ones((batch, k, n), device=device, requires_grad=requires_grad),
        )
        dimensions = tuple(dim if dim == 1 else (dim + 3) // 4 * 4 for dim in (m, k, n))
        actual = self._check_compiled(torch.baddbmm, args, dimensions)
        self.assertEqual(actual.stride(), torch.baddbmm(*args).stride())

    def test_as_strided_consumer(self, device):
        args = (
            torch.arange(24, device=device, dtype=torch.float32).reshape(2, 3, 4),
            torch.ones((2, 3, 4), device=device),
            torch.ones((2, 4, 4), device=device),
        )

        def fn(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2).as_strided((24,), (1,))

        self._check_compiled(fn, args, (4, 4, 4))

    @dtypes(torch.float16, torch.bfloat16)
    def test_beta_zero_ignores_nonfinite_bias(self, device, dtype):
        args = (
            torch.full((2, 3, 9), float("nan"), device=device, dtype=dtype),
            torch.ones((2, 3, 5), device=device, dtype=dtype),
            torch.ones((2, 5, 9), device=device, dtype=dtype),
        )
        for arg in args:
            arg.requires_grad_()

        def fn(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2, beta=0.0, alpha=0.5)

        actual = self._check_compiled(fn, args, (8, 8, 16))
        self.assertTrue(torch.isfinite(actual).all())

    @dtypes(torch.float16)
    def test_alpha_applied_before_half_rounding(self, device, dtype):
        args = (
            torch.full((2, 3, 9), float("inf"), device=device, dtype=dtype),
            torch.full((2, 3, 5), 256.0, device=device, dtype=dtype),
            torch.full((2, 5, 9), 256.0, device=device, dtype=dtype),
        )

        def fn(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2, beta=0.0, alpha=0.125)

        self.assertTrue(torch.isinf(torch.bmm(args[1], args[2])).all())
        actual = self._check_compiled(fn, args, (8, 8, 16))
        self.assertTrue(torch.isfinite(actual).all())

    @parametrize(
        "scalar_name,value", [("beta", float("inf")), ("alpha", float("-inf"))]
    )
    def test_nonfinite_scalars_and_gradients(self, device, scalar_name, value):
        scalars = {scalar_name: value}
        args = (
            torch.ones((9,), device=device, requires_grad=True),
            torch.ones((2, 13, 11), device=device, requires_grad=True),
            torch.ones((2, 11, 9), device=device, requires_grad=True),
        )

        def fn(bias, mat1, mat2):
            return torch.baddbmm(bias, mat1, mat2, **scalars)

        self._check_compiled(fn, args, (16, 12, 12))

    @inductor_config.patch(force_shape_pad=False)
    @parametrize("scalar_name", ["alpha", "beta"])
    def test_dynamic_scalar_fallback(self, device, scalar_name):
        from torch._inductor.fx_passes import pad_mm

        args = (
            torch.ones((9,), device=device),
            torch.ones((2, 13, 11), device=device),
            torch.ones((2, 11, 9), device=device),
            torch.tensor(0.5, device=device),
        )

        def fn(bias, mat1, mat2, scalar):
            return torch.baddbmm(bias, mat1, mat2, **{scalar_name: scalar.item()})

        with unittest.mock.patch.object(
            pad_mm, "is_mm_compute_bound", return_value=True
        ):
            self._check_compiled(fn, args, dimensions=None)

    @fresh_cache()
    @inductor_config.patch(force_shape_pad=False)
    def test_benchmark_scalars_and_caches(self, device):
        from torch._inductor.fx_passes import pad_mm
        from torch._inductor.runtime.caching import encoders
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake_tensor
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.overrides import BaseTorchFunctionMode

        def producers(bias, mat1, mat2, weight):
            return torch.baddbmm(mat1, mat1, weight), mat1 + 1

        with FakeTensorMode():
            mat1 = torch.empty((2, 13, 11), device=device)
            mat2 = torch.empty((2, 11, 9), device=device)
            bias = torch.empty((2, 1, 9), device=device).expand(2, 13, 9)
            weight = torch.empty((2, 11, 11), device=device)
            graph = make_fx(producers)(bias, mat1, mat2, weight).graph
        input_node, _, mat2_node, _ = graph.find_nodes(op="placeholder")
        producer, pointwise = graph.find_nodes(op="output")[0].args[0]
        match = unittest.mock.Mock()
        match.kwargs = dict(
            input=input_node, mat1=producer, mat2=mat2_node, alpha=-0.5, beta=2.0
        )
        self.assertFalse(pad_mm.should_exclude_padding_time(match, "mat1"))
        calls, timed_padding = [], []

        class RecordBaddbmm(BaseTorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.baddbmm:
                    calls.append((args, kwargs))
                if (
                    func is torch.ops.aten.constant_pad_nd
                    and args[0].shape == mat1.shape
                ):
                    timed_padding.append(True)
                return super().__torch_function__(func, types, args, kwargs)

        def bench(fn):
            with RecordBaddbmm():
                fn()
            args, kwargs = calls[-1]
            self.assertEqual(
                kwargs, {name: match.kwargs[name] for name in ("alpha", "beta")}
            )
            self.assertFalse(any(is_fake_tensor(arg) for arg in args))
            if args[1].shape[2] == 11:
                self.assertEqual(args[0].stride(), bias.stride())
                return 2.0
            return 1.0

        op = torch.ops.aten.baddbmm
        with (
            unittest.mock.patch.object(
                pad_mm, "is_mm_compute_bound", return_value=True
            ),
            unittest.mock.patch.object(pad_mm, "get_do_bench", return_value=bench),
        ):
            encoded = []
            for i, (alpha, beta) in enumerate([(0.5, 0.0), (-0.5, 0.0), (0.5, 2.0)]):
                match.kwargs.update(alpha=alpha, beta=beta)
                key = encoders.should_pad_params_encoder(match, mat1, mat2, op, bias)
                self.assertNotIn(key, encoded)
                encoded.append(key)
                self.assertTrue(pad_mm._should_pad(match, mat1, mat2, op, bias))
                self.assertEqual(len(calls), 2 * (i + 1))
                self.assertTrue(pad_mm._should_pad(match, mat1, mat2, op, bias))
                self.assertEqual(len(calls), 2 * (i + 1))
            self.assertTrue(timed_padding)
            timed_padding.clear()
            match.kwargs["mat1"] = pointwise
            self.assertTrue(pad_mm.should_exclude_padding_time(match, "mat1"))
            self.assertTrue(pad_mm._should_pad(match, mat1, mat2, op, bias))
            self.assertEqual(len(calls), 7)
            self.assertFalse(timed_padding)


instantiate_device_type_tests(
    PadBaddbmmTest, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON:
        run_tests()
