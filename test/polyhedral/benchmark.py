import torch
import triton
import triton.testing
from golden_kernel import launch_fused_block
from baseline_example import rms_norm_residual_block, HIDDEN_DIM, SEQ_LEN
from torch._inductor import config
from torch._inductor.utils import fresh_cache


def bench_inductor(x, res, w, polyhedral, warmup=1, rep=25):
    config.polyhedral_fusion = polyhedral
    torch._dynamo.reset()
    with fresh_cache():
        compiled = torch.compile(rms_norm_residual_block, fullgraph=True)
        compiled(x, res, w)
    return triton.testing.do_bench(lambda: compiled(x, res, w), warmup=warmup, rep=rep)


def run_benchmark(warmup=1, rep=25):
    device, dtype = "cuda", torch.bfloat16

    x = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    res = torch.randn(SEQ_LEN, HIDDEN_DIM, device=device, dtype=dtype)
    w = torch.randn(HIDDEN_DIM, device=device, dtype=dtype)

    print(f"Benchmarking Inductor baseline (polyhedral=False, warmup={warmup}, rep={rep})...")
    baseline_ms = bench_inductor(x, res, w, polyhedral=False, warmup=warmup, rep=rep)

    print(f"Benchmarking Inductor fused (polyhedral=True, warmup={warmup}, rep={rep})...")
    fused_ms = bench_inductor(x, res, w, polyhedral=True, warmup=warmup, rep=rep)

    print(f"Benchmarking Golden Kernel (warmup={warmup}, rep={rep})...")
    golden_ms = triton.testing.do_bench(
        lambda: launch_fused_block(x, res, w), warmup=warmup, rep=rep
    )

    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS (triton.testing.do_bench)")
    print(f"Config: SEQ_LEN={SEQ_LEN}, HIDDEN_DIM={HIDDEN_DIM}")
    print(f"Warmup: {warmup}, Repetitions: {rep}")
    print("=" * 60)
    print(f"Inductor baseline (2 kernels):  {baseline_ms:.4f} ms")
    print(f"Inductor polyhedral (1 kernel): {fused_ms:.4f} ms")
    print(f"Golden hand-written (1 kernel): {golden_ms:.4f} ms")
    print("-" * 60)
    print(f"Polyhedral vs baseline:         {baseline_ms / fused_ms:.2f}x")
    print(f"Golden vs baseline:             {baseline_ms / golden_ms:.2f}x")
    print(f"Polyhedral vs golden:           {golden_ms / fused_ms:.2f}x")
    print("=" * 60)

    return baseline_ms, fused_ms, golden_ms


if __name__ == "__main__":
    run_benchmark()
