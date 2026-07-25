"""Benchmark for the consolidated generic sum / mean / nansum reductions.

The changed kernel registrations route the generic `{sum,nansum}_reduction_*`
entry points through the shared value_reduction<SumOp> Metal kernel instead of
the old standalone sum_reduction kernel. The dedicated inner/outer
specializations are intentionally not measured here; they keep their existing
kernels and are not the behavior this PR changes.

Run this on a build WITH the change and on the parent commit (without it) and
compare. The correctness intent is behavior preservation; the performance bar is
no regression, with any speedup coming from the generic tail implementation now
used by value_reduction. amax / amin are included as an unchanged-kernel context
for the existing generic value_reduction path rather than as the primary
pass/fail signal.

Methodology note: the per-shape MPS bench has high run-to-run variance and is
sensitive to machine load/thermal. Run this script on each build back-to-back on
an otherwise-idle machine and compare means across independent measurements;
do not compare a fresh run against a stored baseline taken under different load.
"""

from collections.abc import Callable

import torch
import torch.utils.benchmark as benchmark


# These cases hit the changed generic entry points:
# - full_reduce_1d: large full reduction, two-pass generic kernel
# - middle_dim_contig: contiguous input, single reduced dim that is neither
#   dim=0 nor last dim, so it bypasses the inner/outer specializations
# - multi_dim_noncontig: generic multi-dim + non-contiguous fallback
# - strided_full_reduce: non-contiguous full reduction using strided pass1 +
#   generic pass2
CASES: list[
    tuple[
        str,
        tuple[int, ...],
        int | tuple[int, ...] | None,
        Callable[[torch.Tensor], torch.Tensor],
    ]
] = [
    ("full_reduce_1d", (1 << 24,), None, lambda x: x),
    ("middle_dim_contig", (256, 128, 256), 1, lambda x: x),
    ("multi_dim_noncontig", (16, 64, 16, 64), (1, 3), lambda x: x[:, ::2, :, :]),
    ("strided_full_reduce", (4096, 4096), None, lambda x: x.t()),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# sum/mean/nansum generic entry points migrate onto value_reduction<SumOp>;
# amax/amin show the existing generic value_reduction path for context.
OPS = {
    "sum": torch.sum,
    "mean": torch.mean,
    "nansum": torch.nansum,
    "amax": torch.amax,
    "amin": torch.amin,
}
TRIALS = 5
MIN_RUN_TIME = 0.3


def steady_state():
    print("# steady-state reduction (mean us across independent warm samples)")
    for op_idx, (name, fn) in enumerate(OPS.items()):
        for dtype in DTYPES:
            for case_idx, (case_name, shape, dim, transform) in enumerate(CASES):
                torch.manual_seed(op_idx * 1000 + case_idx)
                x = transform(torch.randn(*shape, device="mps").to(dtype))
                torch.mps.synchronize()
                if dim is None:
                    t = benchmark.Timer(
                        stmt="fn(x); torch.mps.synchronize()",
                        globals={"fn": fn, "x": x, "torch": torch},
                    )
                else:
                    t = benchmark.Timer(
                        stmt="fn(x, dim=dim); torch.mps.synchronize()",
                        globals={"fn": fn, "x": x, "dim": dim, "torch": torch},
                    )
                # MPS execution is asynchronous; synchronizing inside the timed
                # statement makes Timer measure kernel execution rather than
                # enqueue latency.
                trials_us = [
                    t.blocked_autorange(min_run_time=MIN_RUN_TIME).mean * 1e6
                    for _ in range(TRIALS)
                ]
                us = sum(trials_us) / len(trials_us)
                dt = str(dtype).replace("torch.", "")
                print(
                    f"  {case_name:20} {name:7} {dt:9} {str(tuple(x.shape)):18} "
                    f"dim={dim}: {us:9.1f} us "
                    f"(min={min(trials_us):.1f} max={max(trials_us):.1f})"
                )
                del x
                torch.mps.empty_cache()


if __name__ == "__main__":
    steady_state()
