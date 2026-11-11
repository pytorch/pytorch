# Owner(s): ["module: inductor"]
r"""CUDA locality-domain benchmarks. See README.md for setup and usage."""

import re

import pytest

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.utils import counters


pytest.importorskip("pytest_benchmark", reason="requires pytest-benchmark")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module", autouse=True)
def _require_locality_domains():
    if not torch.cuda.green_contexts.is_localization_supported(0):
        pytest.skip("requires CUDA locality domains")


class _TwoSequentialLinears(nn.Module):
    def __init__(self, k, n, p, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(k, n, bias=bias)
        self.fc2 = nn.Linear(n, p, bias=bias)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class _AttentionQKV(nn.Module):
    """Per-head MHA forward with three independent Q/K/V projections.

    The Q/K/V projections become nn.Linear and get localized; the
    rest of the attention (view/transpose/bmm/softmax) runs unchanged
    on the main stream.
    """

    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, batch, seq_len):
        q = (
            self.q(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = q.reshape(batch * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(batch * self.num_heads, seq_len, self.head_dim)
        v = v.reshape(batch * self.num_heads, seq_len, self.head_dim)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim**0.5)
        scores = torch.softmax(scores, dim=-1)
        out = torch.bmm(scores, v)
        out = out.view(batch, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        return out.contiguous().view(batch * seq_len, self.hidden)


class _FFNResidualLN(nn.Module):
    def __init__(self, hidden, ffn_dim):
        super().__init__()
        self.hidden = hidden
        self.ffn = nn.Linear(hidden, ffn_dim, bias=True)
        self.down = nn.Linear(ffn_dim, hidden, bias=False)

    def forward(self, x):
        out = self.ffn(x)
        out = self.down(out) + x
        return F.layer_norm(out, (self.hidden,))


class TestLocalizationBenchmarks:
    """pytest-benchmark suite for the localization path.

    Linear, attention, and FFN cases pair a baseline against a localized
    variant; chain cases benchmark only the localized variant. All cases
    use ``torch.compile`` in ``reduce-overhead`` mode.

    Includes default and forced-Triton GEMM backends.

    Run with::

        pytest benchmarks/localization/test_localization_benchmark.py --benchmark-enable
    """

    @staticmethod
    def _bench_compiled(benchmark, model, args):
        """Compile ``model`` in reduce-overhead mode, warm twice,
        then benchmark a single forward + sync."""
        torch._dynamo.reset()
        counters.clear()
        # Ensure no stray custom pre-pass leaked in from another test.
        with inductor_config.patch(
            {
                "post_grad_custom_pre_pass": None,
            }
        ):
            with torch.inference_mode():
                opt_model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
                opt_model(*args)
                opt_model(*args)
                torch.cuda.synchronize()
                # Benchmarking a skipped capture measures the wrong thing.
                if counters["inductor"]["cudagraph_skips"] != 0:
                    raise AssertionError("expected CUDA Graph capture without skips")

                def _run():
                    opt_model(*args)
                    torch.cuda.synchronize()

                benchmark(_run)

    @staticmethod
    def _compile_localized_capture(model, args, *, force_triton):
        """Compile ``model`` in reduce-overhead mode (matching the other
        benchmarks), capturing the generated wrapper code, and warm twice.
        ``force_triton`` selects Triton-template GEMMs over the default
        ``extern_kernels.mm`` (nvjet) backend.  Returns
        ``(opt_model, generated_code)`` so the caller can assert on numerics
        and IR before handing a run functor to ``benchmark``."""
        from torch._inductor.graph import GraphLowering

        captured: list[str] = []

        def save(code: str) -> None:
            captured.append(code)

        patch = {"post_grad_custom_pre_pass": None}
        if force_triton:
            patch["max_autotune_gemm"] = True
            patch["max_autotune_gemm_backends"] = "TRITON"

        torch._dynamo.reset()
        counters.clear()
        prev = GraphLowering.save_output_code
        GraphLowering.save_output_code = save
        try:
            with inductor_config.patch(patch):
                with torch.inference_mode():
                    opt_model = torch.compile(
                        model, fullgraph=True, mode="reduce-overhead"
                    )
                    opt_model(*args)
                    opt_model(*args)
                    torch.cuda.synchronize()
        finally:
            GraphLowering.save_output_code = prev
        if counters["inductor"]["cudagraph_skips"] != 0:
            raise AssertionError("expected CUDA Graph capture without skips")
        return opt_model, "\n".join(captured)

    def test_simple_matmul_baseline_benchmark(self, benchmark):
        """nn.Linear forward, no localization."""
        m, k, n = 32, 1024, 65536
        model = torch.nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16).eval()
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))

    def _page_aligned_localized(self, benchmark, *, force_triton):
        """Shared body for the two page-aligned localized-matmul cases.

        Shapes are sized so each per-branch matmul output is an exact multiple
        of the locality-domain allocator's page granularity (2 MiB on
        Blackwell): per-branch output = ``m * (n / 2) * 2 = m * n`` bf16 bytes,
        and ``m=32, n=65536`` gives exactly one 2 MiB granularity unit per
        branch -- the alignment the future zero-copy contiguous-split path
        needs.  Compiles in reduce-overhead mode (matching the other
        benchmarks), captures the generated wrapper, asserts numerics vs eager
        and that both branches allocate into their locality-domain pool, then
        benchmarks the steady-state replay.  Returns the generated code so the
        caller can assert backend-specific IR.

        Note: under reduce-overhead the zero-copy static-alloc path is gated
        off (cudagraph_trees rejects out-of-pool storage), so the cat still
        does a pointwise copy; that lifts once cudagraph_trees grows multi-pool
        support.
        """
        m, k, n = 32, 1024, 65536  # m*n = 2 MiB exactly per branch
        eager_model = torch.nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16).eval()
        loc_model = torch.nn.Linear(k, n, bias=False).cuda().to(torch.bfloat16).eval()
        loc_model.load_state_dict(eager_model.state_dict())
        with torch.inference_mode():
            torch.cuda.localize(loc_model)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            ref = eager_model(x)

        opt_model, code = self._compile_localized_capture(
            loc_model, (x,), force_triton=force_triton
        )
        with torch.inference_mode():
            out = opt_model(x)
            torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        # Both branches allocate into their own locality-domain pool.
        pool_count = code.count("use_mem_pool")
        if pool_count != 2:
            raise AssertionError(
                f"expected 2 per-branch use_mem_pool wraps, got {pool_count}"
            )

        def _run():
            opt_model(x)
            torch.cuda.synchronize()

        with torch.inference_mode():
            benchmark(_run)
        return code

    def test_simple_matmul_localized_page_aligned_default_benchmark(self, benchmark):
        """Page-aligned localized nn.Linear with the DEFAULT GEMM backend
        (``extern_kernels.mm`` -> nvjet/cuBLAS).  Companion to the Triton
        variant below: same shapes and same localized fork-join, differing only
        in GEMM backend, so a perf delta between the two isolates backend cost
        from the localization effect."""
        code = self._page_aligned_localized(benchmark, force_triton=False)
        if "extern_kernels.mm" not in code:
            raise AssertionError(
                "expected the default backend to lower the matmul to extern_kernels.mm (nvjet)"
            )

    def test_simple_matmul_localized_page_aligned_triton_benchmark(self, benchmark):
        """Page-aligned localized nn.Linear forcing Triton-template GEMMs
        (``max_autotune_gemm_backends="TRITON"``) instead of nvjet.

        Forcing Triton keeps the matmul's pointer args visible to cupti
        (nvjet hides them), so the per-branch mm can be verified to bind its
        locality-domain pool -- which the default/nvjet variant cannot
        demonstrate."""
        code = self._page_aligned_localized(benchmark, force_triton=True)
        if "extern_kernels.mm" in code:
            raise AssertionError(
                "forced Triton should lower the matmul to a Triton template, not extern_kernels.mm"
            )
        if not re.search("triton_\\w*mm", code):
            raise AssertionError(
                "expected a Triton matmul kernel in the generated code"
            )

    def test_pattern1_attention_baseline_benchmark(self, benchmark):
        """MHA Q/K/V projections + bmm/softmax tail, no localization."""
        batch, seq_len, hidden, num_heads, head_dim = 4, 512, 768, 12, 64
        model = (
            _AttentionQKV(hidden, num_heads, head_dim).cuda().to(torch.bfloat16).eval()
        )
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x, batch, seq_len))

    def test_pattern1_attention_localized_benchmark(self, benchmark):
        """MHA with the three Q/K/V nn.Linears localized."""
        batch, seq_len, hidden, num_heads, head_dim = 4, 512, 768, 12, 64
        model = (
            _AttentionQKV(hidden, num_heads, head_dim).cuda().to(torch.bfloat16).eval()
        )
        with torch.inference_mode():
            torch.cuda.localize(model)
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x, batch, seq_len))

    def test_pattern2_ffn_baseline_benchmark(self, benchmark):
        """FFN (up-proj + down-proj + residual + LN), no localization."""
        batch, seq_len, hidden, ffn_dim = 4, 512, 768, 3072
        model = _FFNResidualLN(hidden, ffn_dim).cuda().to(torch.bfloat16).eval()
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))

    def test_pattern2_ffn_localized_benchmark(self, benchmark):
        """FFN with both up-proj and down-proj localized."""
        batch, seq_len, hidden, ffn_dim = 4, 512, 768, 3072
        model = _FFNResidualLN(hidden, ffn_dim).cuda().to(torch.bfloat16).eval()
        with torch.inference_mode():
            torch.cuda.localize(model)
        x = torch.randn(batch * seq_len, hidden, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))

    @pytest.mark.parametrize(
        ("m", "k", "n", "p"),
        [
            (1, 4096, 32768, 4096),
            (4, 4096, 32768, 4096),
            (16, 4096, 32768, 4096),
            (64, 4096, 32768, 4096),
            (256, 4096, 8192, 4096),
        ],
    )
    def test_chain_localized_benchmark(self, benchmark, m, k, n, p):
        """Two stacked nn.Linears, both localized."""
        model = (
            _TwoSequentialLinears(k, n, p, bias=False).cuda().to(torch.bfloat16).eval()
        )
        with torch.inference_mode():
            torch.cuda.localize(model)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        self._bench_compiled(benchmark, model, (x,))
