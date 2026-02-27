# Owner(s): ["module: inductor"]
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
        @torch.compile()
        def mm(a, b):
            return a @ b

        mm(torch.rand([25, 25], device=GPU_TYPE), torch.rand([25, 25], device=GPU_TYPE))
        local_cache = get_pad_cache().get_local_cache()
        self.assertTrue(len(local_cache) == 2)
        FileCheck().check_count("exclude_pad:False", 2, exactly=True).run(
            repr(local_cache)
        )

        @torch.compile()
        def mm(a, b):
            return (a + 1) @ b

        mm(torch.rand([25, 25], device=GPU_TYPE), torch.rand([25, 25], device=GPU_TYPE))
        local_cache = get_pad_cache().get_local_cache()
        # reuse original base timing
        self.assertTrue(len(local_cache) == 3)

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
            "shape_padding": True,
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


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON:
        run_tests()
