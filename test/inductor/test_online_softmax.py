# Owner(s): ["module: inductor"]

import math
import os
import unittest

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import rmse, same
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, HAS_TRITON


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
USE_LARGE_INPUT = os.environ.get("USE_LARGE_INPUT") == "1" or DO_PERF_TEST
SCALAR_ONLINE_SOFTMAX_CONFIG = {
    "triton.persistent_reductions": False,
    "split_reductions": False,
    "triton.scalar_online_softmax_accumulators": True,
}
requires_nvidia_cuda = unittest.skipUnless(
    GPU_TYPE == "cuda" and torch.version.hip is None,
    "scalar online-softmax accumulators are CUDA-only",
)


def _prepare_softmax(x, dim):
    xmax = x.amax(dim=dim, keepdim=True)
    xsum = (x - xmax).exp().sum(dim=dim, keepdim=True)
    return xmax, xsum


class TestOnlineSoftmax(TestCase):
    def do_test_acc_and_perf(self, op):
        if DO_PERF_TEST:
            N = 32 * 1024
            V = 50304  # padded version for gpt2
        else:
            N, V = 1024, 2048  # small value to avoid OOM in CI

        def f(x):
            return op(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=GPU_TYPE)
        opt_f = torch.compile(f)
        expected = f(x)
        actual = opt_f(x)

        self.assertTrue(same(expected, actual, tol=1e-2))

        if DO_PERF_TEST:
            from triton.testing import do_bench

            eager_ms = do_bench(lambda: f(x))
            opt_ms = do_bench(lambda: opt_f(x))
            print(f"{eager_ms=}")
            print(f"{opt_ms=}")

    def test_softmax(self):
        self.do_test_acc_and_perf(torch.softmax)

    def test_log_softmax(self):
        self.do_test_acc_and_perf(torch.log_softmax)

    @inductor_config.patch(use_fast_math=True)
    def test_prepare_softmax_perf(self):
        self.do_test_acc_and_perf(_prepare_softmax)

    def get_softmax_wrapper(self, V=50304, use_log_softmax=False, device=GPU_TYPE):
        N = 32 * 1024

        @torch.compile
        def f(x):
            if use_log_softmax:
                return torch.log_softmax(x, dim=-1)
            else:
                return torch.softmax(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=device)
        out, source_codes = run_and_get_code(f, x)
        return source_codes[0]

    def test_codegen_3pass_softmax_due_to_disable(self):
        with inductor_config.patch(online_softmax=False):
            wrapper_code = self.get_softmax_wrapper()

        self.assertEqual(wrapper_code.count("for r0_offset in"), 3)

    @parametrize("V", [2048, 50304])
    @parametrize("use_log_softmax", [False, True])
    def test_codegen_online_softmax(self, use_log_softmax, V):
        wrapper_code = self.get_softmax_wrapper(use_log_softmax=use_log_softmax, V=V)

        self.assertEqual(wrapper_code.count("for r0_offset in"), 2)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @parametrize("use_log_softmax", [False, True])
    def test_codegen_online_softmax_unbacked_non_reduction_dim(self, use_log_softmax):
        def f(x, n):
            x = x[: n.item(), :]
            if use_log_softmax:
                return torch.log_softmax(x, dim=-1)
            else:
                return torch.softmax(x, dim=-1)

        x = torch.randn(1024, 2048, dtype=torch.bfloat16, device=GPU_TYPE)
        n = torch.tensor(1024)
        _, source_codes = run_and_get_code(torch.compile(f, fullgraph=True), x, n)
        wrapper_code = "\n".join(source_codes)

        self.assertEqual(wrapper_code.count("for r0_offset in"), 2)
        self.assertTrue("online_softmax_reduce" in wrapper_code)

    def test_no_online_softmax_for_cpu(self):
        code = self.get_softmax_wrapper(V=2048, device="cpu")

        # CPU need an explicit loop across different rows.
        # For GPU, this is parallelized by the hardware.
        self.assertEqual(code.count("for(int64_t"), 4)

    def test_codegen_softmax_persistent_reduction(self):
        """
        Persistent reduction has no for loops.
        """
        wrapper_code = self.get_softmax_wrapper(1024)
        self.assertEqual(wrapper_code.count("for r0_offset in"), 0)

    @inductor_config.patch("triton.persistent_reductions", False)
    def test_sdpa(self):
        """
        Make sure online softmax here does not conflict with the sdpa
        patterns.
        """
        q, k, v = (
            torch.randn((4, 2, 16, 32), device=GPU_TYPE, dtype=torch.bfloat16)
            for _ in range(3)
        )

        def f(q, k, v):
            return (
                torch.matmul(q, k.transpose(-2, -1))
                .div(math.sqrt(k.shape[-1]))
                .softmax(dim=-1)
                .matmul(v)
            )

        opt_f = torch.compile(f)
        ref = f(q, k, v)
        act, (code,) = run_and_get_code(opt_f, q, k, v)
        self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))
        self.assertTrue("aten._scaled_dot_product_" in code)

    @parametrize("nrow", [2, 2048])
    @parametrize("dim", [-1, 0, 1])
    def test_prepare_softmax(self, dim, nrow):
        x = torch.randn(nrow, 2048, dtype=torch.bfloat16, device=GPU_TYPE)
        act, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, dim)
        ref = _prepare_softmax(x, dim)
        self.assertTrue(same(ref, act, tol=1e-2))

        if nrow == 2048 and dim == 0:
            num_kernels = 2
            # split reduction may be triggered depending on the device's SM/CU count.
            # The heuristic in num_splits() in ir.py returns split=1 (no split) when:
            #   numel_hint >= num_sm * 2 * 32
            # When dim=0, numel_hint (output size) = ncol (2048)
            # split is expected only when num_sm > 32
            props = DeviceProperties.create(torch.device(GPU_TYPE))
            num_sm = props.multi_processor_count
            split_not_expected = 2048 >= num_sm * 2 * 32
            if split_not_expected:
                num_kernels = 1

            self.assertTrue(code.count("def triton") >= num_kernels)
        else:
            if nrow == 2 and dim == 0:
                # persistent reduction triggered
                expected_num_loop = 0
            else:
                # A single loop due to online softmax
                expected_num_loop = 1
            self.assertEqual(code.count("for r0_offset in"), expected_num_loop)

    @inductor_config.patch(strict_signed_zero=True)
    def test_prepare_softmax_signed_zero(self):
        def reduce_max(x):
            return x.amax(dim=-1, keepdim=True)

        x = torch.zeros(2, 2048, device=GPU_TYPE)
        x[0, 1::2] = -0.0
        x[1, ::2] = -0.0

        ref = _prepare_softmax(x, -1)
        ref_max = torch.compile(reduce_max, fullgraph=True)(x)
        act, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, -1)

        self.assertIn("online_softmax_reduce", code)
        self.assertEqual(ref_max.view(torch.int32), act[0].view(torch.int32))
        self.assertEqual(ref[1], act[1])

    def test_prepare_softmax_after_partitioning(self):
        from torch._dynamo.backends.common import aot_autograd
        from torch._functorch.aot_autograd import make_boxed_func
        from torch._functorch.partitioners import min_cut_rematerialization_partition
        from torch._inductor.decomposition import select_decomp_table
        from torch._inductor.fx_passes import post_grad
        from torch._inductor.fx_passes.joint_graph import joint_graph_passes
        from torch._inductor.pattern_matcher import (
            fwd_only,
            PatternMatcherPass,
            register_replacement,
        )

        def apply_online_softmax_in_joint_graph(gm):
            patterns = PatternMatcherPass(pass_name="test_joint_online_softmax")
            register_replacement(
                post_grad.prepare_softmax_pattern,
                post_grad.prepare_softmax_replacement,
                [torch.empty(4, 8)],
                fwd_only,
                patterns,
                scalar_workaround={"dim": -1},
                extra_check=post_grad.prepare_softmax_extra_check,
            )
            return patterns.apply(gm.graph)

        def saved_activation_bytes(fw_module, num_fwd_outputs):
            (output_node,) = fw_module.graph.find_nodes(op="output")
            saved_values = output_node.args[0][num_fwd_outputs:]
            total = 0
            for node in saved_values:
                if not isinstance(node, torch.fx.Node):
                    continue
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
            return total

        def compile_and_capture_saved_bytes(apply_joint_online_softmax):
            captured = {}

            def partition_fn(gm, joint_inputs, *, num_fwd_outputs, **kwargs):
                joint_graph_passes(gm)
                if apply_joint_online_softmax:
                    captured["joint_online_softmax_count"] = (
                        apply_online_softmax_in_joint_graph(gm)
                    )
                    gm.graph.lint()
                    gm.recompile()
                captured["joint_graph_code"] = gm.code
                fw_module, bw_module = min_cut_rematerialization_partition(
                    gm,
                    joint_inputs,
                    num_fwd_outputs=num_fwd_outputs,
                    **kwargs,
                )
                captured["saved_activation_bytes"] = saved_activation_bytes(
                    fw_module, num_fwd_outputs
                )
                return fw_module, bw_module

            def compiler(gm, example_inputs):
                return make_boxed_func(gm.forward)

            backend = aot_autograd(
                fw_compiler=compiler,
                bw_compiler=compiler,
                partition_fn=partition_fn,
                decompositions=select_decomp_table(),
            )

            def fn(q, k, target):
                scores = q @ k.transpose(-2, -1)
                probs = scores.softmax(dim=-1)
                return (probs * target).sum() + scores.sin().sum()

            B, S, D = 2, 128, 16
            args = (
                torch.randn(
                    B, S, D, device=GPU_TYPE, dtype=torch.float16, requires_grad=True
                ),
                torch.randn(
                    B, S, D, device=GPU_TYPE, dtype=torch.float16, requires_grad=True
                ),
                torch.randn(
                    B, S, S, device=GPU_TYPE, dtype=torch.float16, requires_grad=True
                ),
            )

            torch._dynamo.reset()
            loss = torch.compile(fn, backend=backend, fullgraph=True)(*args)
            torch.autograd.grad(loss, args)
            return captured

        current = compile_and_capture_saved_bytes(apply_joint_online_softmax=False)
        joint_online = compile_and_capture_saved_bytes(apply_joint_online_softmax=True)

        self.assertNotIn("prepare_softmax_online", current["joint_graph_code"])
        self.assertEqual(joint_online["joint_online_softmax_count"], 1)
        self.assertIn("prepare_softmax_online", joint_online["joint_graph_code"])
        self.assertLessEqual(
            joint_online["saved_activation_bytes"],
            current["saved_activation_bytes"],
        )

    @parametrize("strict_signed_zero", [False, True])
    @inductor_config.patch("triton.persistent_reductions", False)
    def test_split_reduction(self, strict_signed_zero):
        """
        Split online_softmax_reduce into partial max/sum tuples and combine
        the partials with another online_softmax_reduce.
        """
        # tensor shape to trigger split reduction
        x = torch.randn(1, 2**20 + 13, dtype=torch.bfloat16, device=GPU_TYPE)
        ref = torch.softmax(x, dim=-1)
        with inductor_config.patch(strict_signed_zero=strict_signed_zero):
            act, (code,) = run_and_get_code(torch.compile(torch.softmax), x, dim=-1)
        self.assertTrue(torch.allclose(ref, act, atol=1e-3, rtol=1e-3))
        self.assertTrue(code.count("def triton") >= 2)
        self.assertTrue("online_softmax_reduce" in code)
        self.assertTrue("online_softmax_combine_with_sum" in code)

    def test_kl_div_log_softmax_backward_split_reduction(self):
        logits = torch.randn(
            1, 2**20, dtype=torch.float32, device=GPU_TYPE, requires_grad=True
        )
        targets = F.softmax(torch.randn_like(logits), dim=-1)
        ref_logits = logits.detach().clone().requires_grad_()
        ref_targets = targets.detach().clone()

        def f(logits, targets):
            return F.kl_div(
                F.log_softmax(logits, dim=-1), targets, reduction="batchmean"
            )

        ref = f(ref_logits, ref_targets)
        ref.backward()

        opt_f = torch.compile(f)
        act, codes = run_and_get_code(opt_f, logits, targets)
        act.backward()
        code = "\n".join(codes)

        self.assertEqual(ref, act)
        self.assertEqual(ref_logits.grad, logits.grad)
        self.assertTrue("online_softmax_reduce" in code)
        self.assertTrue("online_softmax_combine_with_sum" in code)

    @parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
    def test_prepare_softmax_acc_with_fp64(self, dtype):
        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        ref_fp64 = _prepare_softmax(x.to(dtype=torch.float64), dim=-1)
        ref = _prepare_softmax(x, dim=-1)
        res, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, dim=-1)
        self.assertTrue("online_softmax_reduce" in code)

        # Max should be exactly equal
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[0].to(dtype=torch.float64), ref_fp64[0])

        ref_error = rmse(ref_fp64[1], ref[1]).item()
        res_error = rmse(ref_fp64[1], res[1]).item()

        # My local tests even shows a smaller res_error:
        #   ref_error=2.1065, res_error=2.1028
        # for bf16
        #   ref_error=0.2611, res_error=0.2609
        # for fp16
        #   ref_error=0.0001, res_error=0.0001
        # for fp32
        print(f"{ref_error=:.4f}, {res_error=:.4f}")

        self.assertTrue(
            res_error < ref_error + 0.1
        )  # Is this good enough to make CI stable

    @parametrize("fn", [torch.log_softmax, torch.softmax])
    @parametrize("dtype", [torch.bfloat16, torch.half, torch.float32])
    def test_softmax_acc_with_fp64(self, dtype, fn):
        if USE_LARGE_INPUT:
            M, N = 32768, 50257
        else:
            M, N = 1024, 2048

        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        ref_fp64 = fn(x.to(dtype=torch.float64), dim=-1)
        ref = fn(x, dim=-1)
        res, (code,) = run_and_get_code(torch.compile(fn), x, dim=-1)
        self.assertTrue("online_softmax_reduce" in code)

        ref_error = rmse(ref_fp64, ref).item()
        res_error = rmse(ref_fp64, res).item()

        # For torch.softmax,
        # I get almost 0 for ref_error/res_error for all 3 dtypes. It's because
        # each value is very small since each row add up to 1.0
        #
        # For torch.log_softmax
        #   ref_error=0.0180399032, res_error=0.0180399031
        # for bf16
        #   ref_error=0.0022548872, res_error=0.0022548872
        # for fp16
        #   ref_error=0.0000003744, res_error=0.0000003748
        # for fp32
        print(f"{ref_error=:.10f}, {res_error=:.10f}")

        self.assertTrue(
            res_error < ref_error + 0.1
        )  # Is this good enough to make CI stable

    def test_softmin(self):
        """
        The rnumel==1 kind of reduction should be unrolled.
        """

        def f(x):
            return F.softmin(x, dim=0)

        x = torch.randn(1, device=GPU_TYPE)
        ref = f(x)
        act, (code,) = run_and_get_code(torch.compile(f), x)
        self.assertTrue(torch.allclose(ref, act))
        self.assertTrue("online_softmax_reduce" not in code)

    def test_causal_mask(self):
        def f(x):
            return x.softmax(dim=-1)

        x = torch.randn(2048, 2048, device=GPU_TYPE)
        mask = torch.tril(torch.ones(2048, 2048, device=GPU_TYPE))
        x.masked_fill_(mask == 0, float("-inf"))

        ref = f(x)
        act = torch.compile(f)(x)
        self.assertTrue(not ref.isnan().any())
        self.assertTrue(not act.isnan().any())
        self.assertTrue(torch.allclose(ref, act))

    def test_tb_speech_transformer_attn(self):
        """
        This is an example extracted from speech_transformer.
        Since online softmax use the max from partial elements of an entire
        row, if the input contains '-inf', it's possible that the
        max of those partial elements is '-inf' even if the entire row
        has non '-inf' value. In this cause, online softmax will need
        do things like 'float(-inf) - float(-inf)' which becomes 'nan'.
        We fixed this by interpreting 'float(-inf) - float(-inf)' as 0
        if we found both operands are 'float(-inf)'.
        """
        torch.manual_seed(1337)

        def f(x, mask):
            x = torch.where(mask, float("-inf"), x)
            xmax = x.amax(dim=-1, keepdim=True)
            xsum = (x - xmax).exp().sum(dim=-1, keepdim=True)
            return xsum

        x = torch.randn(8, 10, 22, 204, device=GPU_TYPE)
        mask = torch.randint(0, 2, (10, 204), device=GPU_TYPE) == 0
        mask = mask.view(1, 10, 1, 204)

        ref = f(x, mask)
        act = torch.compile(f)(x, mask)
        self.assertTrue(not ref.isnan().any())
        self.assertTrue(not act.isnan().any())
        self.assertTrue(torch.allclose(ref, act))

    @inductor_config.patch(split_reductions=False)
    def test_3d_tiled_online_softmax(self):
        def f(x, y):
            return (x * y).softmax(dim=-1)

        M, N, K = 32, 8, 1024

        x = torch.randn(K, N, M, device=GPU_TYPE).permute(2, 1, 0)
        y = torch.randn(K, M, N, device=GPU_TYPE).permute(1, 2, 0)

        opt_f = torch.compile(f)
        torch.testing.assert_close(f(x, y), opt_f(x, y), atol=1e-3, rtol=1e-3)

    @parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_nan_propagation(self, dtype):
        """
        The softmax-internal max uses fmax (non-NaN-propagating) for
        performance, but NaN must still propagate to the final output
        because the original input flows through exp(x - xmax).
        Place NaN at the beginning, middle, and end of separate rows.

        This is Triton-only because fmax is only implemented in the
        Triton persistent reduction path.
        """
        if not HAS_TRITON:
            self.skipTest("requires triton")

        M, N = 4, 1024
        x = torch.randn(M, N, device=GPU_TYPE, dtype=dtype)

        x[0, 0] = float("nan")
        x[1, N // 2] = float("nan")
        x[2, N - 1] = float("nan")
        # row 3 has no NaN

        ref = torch.softmax(x, dim=-1)
        act, (code,) = run_and_get_code(torch.compile(torch.softmax), x, dim=-1)

        self.assertIn("fmax2", code)

        # Rows with NaN input must produce all-NaN output
        for row in range(3):
            self.assertTrue(
                ref[row].isnan().all(),
                f"eager row {row} should be all NaN",
            )
            self.assertTrue(
                act[row].isnan().all(),
                f"compiled row {row} should be all NaN",
            )

        # Row without NaN must match exactly
        self.assertFalse(act[3].isnan().any())
        torch.testing.assert_close(ref[3], act[3])


@requires_nvidia_cuda
@inductor_config.patch(SCALAR_ONLINE_SOFTMAX_CONFIG)
@instantiate_parametrized_tests
class TestScalarOnlineSoftmax(TestCase):
    """Per-row max/sum accumulators for large non-persistent online softmax."""

    MARKER = "online_softmax_reduce_scalar_combine"
    COMBO_KERNEL_CONFIG = {
        "combo_kernels": True,
        "combo_kernel_per_subkernel_blocks": True,
        "combo_kernel_max_distance": -1,
        "combo_kernel_peak_memory_increase_gb": None,
        "combo_kernel_peak_memory_pct_threshold": None,
    }

    def check_codegen(self, fn, *args, uses_scalar=True, rtol=1e-3, atol=1e-3):
        act, (code,) = run_and_get_code(torch.compile(fn), *args)
        self.assertEqual(fn(*args), act, rtol=rtol, atol=atol)
        if uses_scalar:
            self.assertIn(self.MARKER, code)
        else:
            self.assertNotIn(self.MARKER, code)
        return act, code

    @inductor_config.patch("triton.scalar_online_softmax_accumulators", False)
    def test_disabled(self):
        x = torch.randn(1024, 8192, device=GPU_TYPE)
        _, code = self.check_codegen(_prepare_softmax, x, -1, uses_scalar=False)
        self.assertIn("online_softmax_combine(", code)

    @parametrize("op", (torch.softmax, torch.log_softmax))
    def test_softmax_ops(self, op):
        x = torch.randn(4, 8193, device=GPU_TYPE)
        self.check_codegen(lambda t: op(t, dim=-1), x)

    def test_50k_reduction_bucket(self):
        storage = torch.randn(4, 50272, dtype=torch.bfloat16, device=GPU_TYPE)
        _, code = self.check_codegen(
            _prepare_softmax, storage[:, :50265], -1, rtol=1e-2, atol=1e-2
        )
        self.assertIn("AutotuneHint.SCALAR_ONLINE_SOFTMAX", code)

    @parametrize("reduction_numel,uses_scalar", [(4096, False), (4097, True)])
    def test_reduction_size_threshold(self, reduction_numel, uses_scalar):
        x = torch.randn(2, reduction_numel, device=GPU_TYPE)
        self.check_codegen(_prepare_softmax, x, -1, uses_scalar=uses_scalar)

    def test_skips_outer_reduction(self):
        x = torch.randn(8192, 128, device=GPU_TYPE)
        _, code = self.check_codegen(_prepare_softmax, x, 0, uses_scalar=False)
        self.assertIn("online_softmax_combine(", code)

    @parametrize("num_inputs,uses_scalar", [(3, True), (4, False)])
    def test_read_limit(self, num_inputs, uses_scalar):
        def f(*args):
            value = args[0]
            for arg in args[1:]:
                value = value + arg
            return _prepare_softmax(value, -1)

        args = [torch.randn(128, 8192, device=GPU_TYPE) for _ in range(num_inputs)]
        self.check_codegen(f, *args, uses_scalar=uses_scalar)

    def test_allows_fused_gather(self):
        def f(logits, target):
            xmax, xsum = _prepare_softmax(logits, -1)
            target_logit = logits.gather(-1, target[:, None])
            return (xmax + xsum.log() - target_logit).squeeze(-1)

        logits = torch.randn(128, 8192, device=GPU_TYPE)
        target = torch.randint(0, 8192, (128,), device=GPU_TYPE)
        self.check_codegen(f, logits, target)

    @parametrize("materialized_outputs,uses_scalar", [(1, True), (2, False)])
    def test_materialized_output_limit(self, materialized_outputs, uses_scalar):
        def f(x, bias):
            logits = (x + bias).to(torch.bfloat16)
            xmax, xsum = _prepare_softmax(logits.float(), -1)
            if materialized_outputs == 1:
                return logits, xmax + xsum.log()
            return logits, (logits.float() - xmax - xsum.log()).to(torch.bfloat16)

        x = torch.randn(128, 8192, dtype=torch.bfloat16, device=GPU_TYPE)
        bias = torch.randn(8192, device=GPU_TYPE)
        self.check_codegen(f, x, bias, uses_scalar=uses_scalar, rtol=1e-2, atol=1e-2)

    def test_dynamic_batch_uses_scalar_path(self):
        torch._dynamo.reset()
        counter = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        opt_f = torch.compile(_prepare_softmax, backend=counter)
        x = torch.randn(4, 8193, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)
        act, (code,) = run_and_get_code(opt_f, x, -1)
        self.assertEqual(_prepare_softmax(x, -1), act, rtol=1e-3, atol=1e-3)
        self.assertIn(self.MARKER, code)

        x = torch.randn(7, 8193, device=GPU_TYPE)
        self.assertEqual(_prepare_softmax(x, -1), opt_f(x, -1), rtol=1e-3, atol=1e-3)
        self.assertEqual(counter.frame_count, 1)

    def test_unmasked_loop(self):
        # xnumel 1 and rnumel a multiple of the largest R0_BLOCK need no masks,
        # so the combine receives a literal True mask.
        x = torch.randn(1, 65536, device=GPU_TYPE)
        _, code = self.check_codegen(_prepare_softmax, x, -1)
        self.assertRegex(code, r"scalar_combine\(\s+\S+, \S+, \S+, True, 1,")

    def test_combo_kernel_uses_vector_path(self):
        def f(x, y):
            return (*_prepare_softmax(x, -1), *_prepare_softmax(y, -1))

        x = torch.randn(4, 8192, device=GPU_TYPE)
        y = torch.randn(4, 8192, device=GPU_TYPE)
        with inductor_config.patch(self.COMBO_KERNEL_CONFIG):
            act, codes = run_and_get_code(torch.compile(f), x, y)

        self.assertEqual(f(x, y), act, rtol=1e-3, atol=1e-3)
        code = "\n".join(codes)
        self.assertIn("combo_grid_meta", code)
        self.assertNotIn(self.MARKER, code)
        self.assertIn("online_softmax_combine(", code)

    def test_skips_extra_reductions(self):
        def f(x):
            xmax, xsum = _prepare_softmax(x, -1)
            return xmax, xsum, (x - xmax).sum(dim=-1, keepdim=True)

        x = torch.randn(128, 8192, device=GPU_TYPE)
        _, code = self.check_codegen(f, x, uses_scalar=False)
        self.assertIn("online_softmax_combine(", code)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    def test_nan_and_inf_rows(self, dtype):
        rows, cols = 7, 8193
        x = torch.randn(rows, cols, device=GPU_TYPE, dtype=dtype)
        x[0, 0] = float("nan")
        x[1, cols // 2] = float("nan")
        x[2, cols - 1] = float("nan")
        x[3].fill_(float("-inf"))
        x[4].fill_(float("-inf"))
        x[4, 0] = float("inf")
        x[5].fill_(float("-inf"))
        x[5, cols // 2] = 2.0

        ref = _prepare_softmax(x, -1)
        act, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, -1)
        self.assertIn(self.MARKER, code)
        # Eager returns NaN for the sum of an all -inf row; inductor returns the count.
        eager_rows = torch.tensor([0, 1, 2, 4, 5, 6], device=GPU_TYPE)
        for expected, actual in zip(ref, act):
            self.assertTrue(actual[:3].isnan().all())
            self.assertEqual(expected[eager_rows], actual[eager_rows], equal_nan=True)
        self.assertTrue(act[0][3].isneginf().all())
        self.assertEqual(act[1][3], torch.full_like(act[1][3], cols))

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_acc_with_fp64(self, dtype):
        x = torch.randn(128, 8193, device=GPU_TYPE, dtype=dtype)
        ref_fp64 = _prepare_softmax(x.to(torch.float64), -1)
        ref = _prepare_softmax(x, -1)
        act, (code,) = run_and_get_code(torch.compile(_prepare_softmax), x, -1)
        self.assertIn(self.MARKER, code)
        self.assertEqual(ref[0], act[0])
        ref_error = rmse(ref_fp64[1], ref[1]).item()
        act_error = rmse(ref_fp64[1], act[1]).item()
        self.assertLessEqual(act_error, ref_error + 0.1)

    @inductor_config.patch(strict_signed_zero=True)
    def test_signed_zero(self):
        def reduce_max(x):
            return x.amax(dim=-1, keepdim=True)

        x = torch.zeros(2, 8193, device=GPU_TYPE)
        x[0, 1::2] = -0.0
        x[1, ::2] = -0.0
        ref_max = torch.compile(reduce_max, fullgraph=True)(x)
        act, _ = self.check_codegen(_prepare_softmax, x, -1)
        self.assertEqual(ref_max.view(torch.int32), act[0].view(torch.int32))

    @inductor_config.patch({"triton.max_tiles": 3, "triton.prefer_nd_tiling": True})
    def test_3d_tiling(self):
        def f(x, y):
            return (x * y).softmax(dim=-1)

        M, N, K = 32, 8, 8193
        x = torch.randn(M, N, K, device=GPU_TYPE)
        y = torch.randn(N, M, K, device=GPU_TYPE).permute(1, 0, 2)
        _, code = self.check_codegen(f, x, y)
        self.assertIn("YBLOCK", code)


instantiate_parametrized_tests(TestOnlineSoftmax)

if __name__ == "__main__":
    if IS_LINUX and HAS_GPU and HAS_TRITON:
        run_tests()
