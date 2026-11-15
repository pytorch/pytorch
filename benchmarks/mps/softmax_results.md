# MPS softmax: collapse correctness and performance

Measured on 2026-11-15 on an Apple M4 with 24 GiB RAM, macOS 15.6.1.
The baseline is **local `main` at
`760f1c8da40f395aca97825bc2e387d7c5afaab1`**, not the latest remote branch.
The candidate is based on `ffe084918476d7f60baf8bb51ebc927c2cc8f197` plus
the accompanying Metal and test changes. Its kernel SHA-256 is
`cb9e8317fac79d0ad1947a15debd47356f542959c1842d387ed82ee3469ca848`.

The FP32 latency gate passes **40/40 operator/shape combinations**,
with a worst pooled-median change of **+4.20%** against the 5% per-shape
limit. Memory results cover 70 probes per operator. These measurements are
specific to this machine; no performance claim is made for other M-series GPUs.

## Why the numerical change is needed

After subtracting a finite row maximum, one exponential is exactly one. Let
`t` be the sum of the remaining exponentials. Adding that one too early can
round `1 + t` to one and erase a representable correction to the dominant
probability or log probability.

Both forward operators now retain `t` separately. Log-softmax uses `log1p(t)`.
Softmax computes `inv = 1 / (1 + t)` and uses `fma(-t, inv, 1)` when `t < 1`;
otherwise it uses `inv`. This preserves the small probability complement in
the final rounding. It does not prevent probabilities from rounding to one
when that is the correct result for the output dtype.

Most reductions accumulate non-maximum exponentials and count finite maxima
separately, then form `t = nonmax_sum + (max_count - 1)`. This removes exactly
one unit term, including when maxima tie, without tracking a winning index
through every reduction. Split partials retain the same tail representation;
the final reduction rescales partial tails and restores all but one omitted
maximum. Existing nonfinite behavior and the log-softmax backward `expm1`
correction are preserved.

## Keeping the kernel simple and fast

`softmax_row` has no `#pragma unroll` directives. Its general path shares the
max, sum, and output loops across operations. A short forward row keeps at most
two inputs and exponentials per lane, omits one winning maximum, and reuses the
cached exponentials for softmax output. The scalar operation overload supports
this reuse without changing the normalization formula.

The contiguous split partial kernel caches at most eight input floats per
thread. Constant loop bounds let the compiler optimize that storage; larger
chunks and backward operations use the ordinary loops. Dynamic indexing of a
bounded cache was measured and rejected because it slowed split rows. Neither
cache size depends on the M4's core count. Host dispatch and tensor allocations
are unchanged by this patch.

## Latency

The exact benchmark source from `ffe08491847` was saved before switching to
main, and used for both revisions. Both builds used:

```sh
source venv/bin/activate
USE_DISTRIBUTED=0 CMAKE_BUILD_TYPE=Release pip install -e . -v --no-build-isolation
```

Inputs and output buffers are preallocated, with one shape resident at a time.
Each process runs five warmups and eight trials per operator/shape, alternating
operator order. A pilot selects 1-300 calls per trial targeting 200 ms. The first
trial is discarded as specified before measurement; all remaining trials are
retained. The table pools two baseline runs (14 samples) and three final
candidate runs (21 samples). Positive percentages are regressions.
Some individual runs exceeded 5%; the gate applies to pooled medians.

MPS event intervals include queue gaps and are not isolated shader timings.
Wall timings are saved separately. Sequential runs showed timing variation:
an earlier candidate passed 39/40 after two runs, with one shape at
+5.05%. It was not rounded into a pass. The final source was rebuilt and measured
in three full runs; no shape or unfavorable retained trial was removed.

All times below are microseconds, including large tensors whose times exceed
100,000 microseconds.

| Case | Shape | Dim | Softmax main | Candidate | Change | Log-softmax main | Candidate | Change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `(4, 499, 46)` | 2 | 6.548 | 5.246 | -19.88% | 6.263 | 5.933 | -5.27% |
| 1 | `(1, 201088)` | 1 | 13.047 | 12.697 | -2.69% | 12.837 | 12.577 | -2.02% |
| 2 | `(5, 51866)` | 1 | 15.789 | 15.038 | -4.76% | 15.074 | 14.788 | -1.90% |
| 3 | `(1, 151936)` | 1 | 11.309 | 10.626 | -6.05% | 10.902 | 10.384 | -4.75% |
| 4 | `(64, 1000)` | 1 | 9.309 | 9.700 | +4.20% | 9.249 | 9.264 | +0.16% |
| 5 | `(4, 129280)` | 1 | 35.311 | 31.516 | -10.75% | 33.930 | 31.166 | -8.15% |
| 6 | `(4, 151936)` | 1 | 46.153 | 43.049 | -6.73% | 46.501 | 43.218 | -7.06% |
| 7 | `(4096, 30522)` | 1 | 18499.083 | 18326.522 | -0.93% | 18368.591 | 18306.910 | -0.34% |
| 8 | `(2, 150, 512, 512)` | 1 | 10701.635 | 10852.842 | +1.41% | 10716.121 | 10791.305 | +0.70% |
| 9 | `(8, 154880)` | 1 | 105.777 | 106.236 | +0.43% | 106.067 | 107.215 | +1.08% |
| 10 | `(4, 499, 2341)` | 1 | 515.525 | 518.191 | +0.52% | 519.710 | 517.757 | -0.38% |
| 11 | `(1024, 50257)` | 1 | 9080.544 | 8611.649 | -5.16% | 9080.677 | 8651.998 | -4.72% |
| 12 | `(2048, 128256)` | 1 | 46004.958 | 43463.000 | -5.53% | 45807.538 | 43508.867 | -5.02% |
| 13 | `(2048, 129280)` | 1 | 45912.108 | 43557.300 | -5.13% | 45155.825 | 43774.975 | -3.06% |
| 14 | `(2048, 151936)` | 1 | 52599.328 | 51724.542 | -1.66% | 52942.000 | 51429.896 | -2.86% |
| 15 | `(8192, 151936)` | 1 | 214847.208 | 205457.250 | -4.37% | 216320.000 | 205453.208 | -5.02% |
| 16 | `(2048, 154880)` | 1 | 54233.714 | 52300.354 | -3.56% | 54176.391 | 52218.594 | -3.61% |
| 17 | `(1024, 201088)` | 1 | 34702.142 | 33917.729 | -2.26% | 34725.646 | 33855.687 | -2.51% |
| 18 | `(4, 1023, 151936)` | 2 | 105396.792 | 102095.500 | -3.13% | 104967.250 | 102078.187 | -2.75% |
| 19 | `(4, 151936, 128)` | 1 | 9546.950 | 9177.527 | -3.87% | 9486.558 | 9172.803 | -3.31% |

## Memory

Each operator ran in its own fresh process. The 70 probes consist of the 20
model shapes, a warmup, two sweeps of 24 changing shapes, and a large final
probe. Every probe records before/live/deleted/clean snapshots, allocator
peaks, and driver memory. Driver snapshots are not driver peaks. Per-thread
register/cache storage and threadgroup arrays are not separately quantified by
these allocator measurements.

| Operator | Probes | Largest allocator peak (MiB) | Retained driver range (MiB) | All snapshot fields match main |
| --- | ---: | ---: | --- | --- |
| softmax | 70 | 9728 | 0.453125-0.468750 | Yes |
| log_softmax | 70 | 9728 | 0.453125-0.468750 | Yes |

## Correctness and reproduction

The final build passed 1,017 softmax-related MPS tests, with nine existing
expected failures, and 120 model checks across both operators and float32,
float16, and bfloat16. The collapse regressions check exact dominant softmax
probabilities, tiny log-softmax tails, tied maxima, nonfinite inputs, and the
row, split, generic, and strided paths. Main failed 84 of the 168 new forward
collapse cases used as a negative control.

An additional local shader check passed 864 combinations of both index widths,
three dtypes, nine row widths, four input patterns, and all four forward/backward
operations. It compared direct row-kernel calls with the rebuilt public
operators. The u64 checks use small allocations; they do not establish behavior
at offsets beyond 32 bits.

Run the committed checks and benchmarks from the repository root:

```sh
source venv/bin/activate
python test/test_mps.py -k softmax
python benchmarks/mps/test_softmax.py
python benchmarks/mps/log_softmax.py --output agent_space/candidate_latency.json
python benchmarks/mps/log_softmax_memory.py --op softmax --output agent_space/candidate_softmax_memory.json
python benchmarks/mps/log_softmax_memory.py --op log_softmax --output agent_space/candidate_log_softmax_memory.json
spin quicklint -a
```

Save the benchmark source before changing revisions, rebuild main with the same
configuration, and repeat the benchmark commands in fresh processes. For the
measurements recorded here, the comparison command was:

```sh
python benchmarks/mps/compare_softmax.py \
  --baseline agent_space/softmax_parity/main_latency.json agent_space/softmax_parity/main_latency_repeat.json \
  --candidate agent_space/softmax_parity/final_latency.json agent_space/softmax_parity/final_latency_repeat.json agent_space/softmax_parity/final_latency_third.json \
  --output agent_space/softmax_parity/final_comparison.json
```

Raw trial data, memory snapshots, source snapshots, build/test logs, and local
shader diagnostics are retained in `agent_space/softmax_parity/` in the measuring
workspace. That scratch directory is git-ignored. The benchmark source, gate,
model checks, numerical regressions, and this report are committed.
