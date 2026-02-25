"""Microbenchmark for index_add_ targeting the indexFuncLargeIndex CUDA kernel.

Focused on real workload shapes from IG ranking models:
  - self/source: BFloat16 [~4M, 128], dim=0, index: int64 [~4M]

Run via:
    buck2 run //caffe2/benchmarks/operator_benchmark:bench_index_add_large @//mode/dev-nosan 2>/dev/null
"""

import time

import torch


def _bench(fn, *, warmup: int = 50, iters: int = 200) -> float:
    """Return average latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> None:
    device = "cuda"
    torch.manual_seed(42)

    # ------------------------------------------------------------------ #
    # Real workload shape from IG ranking models                          #
    # ------------------------------------------------------------------ #
    real_workload_configs = [
        # (dst_shape, dim, num_index, src_shape, dtype, description)
        (
            (4036236, 128),
            0,
            4036236,
            (4036236, 128),
            torch.bfloat16,
            "REAL bf16 [4036236,128] dim=0 nIdx=4036236",
        ),
    ]

    print("=" * 72)
    print("Real workload: index_add_  (IG ranking)")
    print("=" * 72)
    for dst_shape, dim, num_idx, src_shape, dtype, desc in real_workload_configs:
        dst = torch.zeros(dst_shape, dtype=dtype, device=device)
        index = torch.randint(0, dst_shape[dim], (num_idx,), device=device, dtype=torch.long)
        src = torch.randn(src_shape, dtype=dtype, device=device)

        us = _bench(lambda: dst.index_add_(dim, index, src))
        print(f"  {desc:55s}  {us:9.1f} us")

    # ------------------------------------------------------------------ #
    # Sweep around the real workload to understand scaling                 #
    # ------------------------------------------------------------------ #
    sweep_configs = [
        # (num_rows, embed_dim, description)
        (1000000, 128, "1M x 128"),
        (2000000, 128, "2M x 128"),
        (4000000, 128, "4M x 128 (≈real)"),
        (8000000, 128, "8M x 128"),
        (4000000, 64, "4M x 64"),
        (4000000, 256, "4M x 256"),
        (4000000, 512, "4M x 512"),
    ]

    print()
    print("=" * 72)
    print("Sweep: index_add_  (bf16, dim=0, CUDA)")
    print("=" * 72)
    for num_rows, embed_dim, desc in sweep_configs:
        try:
            dst = torch.zeros(num_rows, embed_dim, dtype=torch.bfloat16, device=device)
            index = torch.randint(0, num_rows, (num_rows,), device=device, dtype=torch.long)
            src = torch.randn(num_rows, embed_dim, dtype=torch.bfloat16, device=device)

            us = _bench(lambda: dst.index_add_(dim, index, src))
            total_bytes = num_rows * embed_dim * 2 * 2  # read src + write dst, 2B per bf16
            bw_gbps = total_bytes / (us * 1e-6) / 1e9
            print(f"  {desc:30s}  {us:9.1f} us  ({bw_gbps:6.1f} GB/s eff.)")
        except torch.cuda.OutOfMemoryError:
            print(f"  {desc:30s}       OOM")

    # ------------------------------------------------------------------ #
    # float32 comparison                                                   #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 72)
    print("Sweep: index_add_  (float32, dim=0, CUDA)")
    print("=" * 72)
    for num_rows, embed_dim, desc in sweep_configs[:4]:
        try:
            dst = torch.zeros(num_rows, embed_dim, dtype=torch.float32, device=device)
            index = torch.randint(0, num_rows, (num_rows,), device=device, dtype=torch.long)
            src = torch.randn(num_rows, embed_dim, dtype=torch.float32, device=device)

            us = _bench(lambda: dst.index_add_(dim, index, src))
            total_bytes = num_rows * embed_dim * 4 * 2  # 4B per float32
            bw_gbps = total_bytes / (us * 1e-6) / 1e9
            print(f"  {desc:30s}  {us:9.1f} us  ({bw_gbps:6.1f} GB/s eff.)")
        except torch.cuda.OutOfMemoryError:
            print(f"  {desc:30s}       OOM")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
