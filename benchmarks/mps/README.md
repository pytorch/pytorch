# MPS softmax benchmarks

See [the M4 comparison report](softmax_results.md) for per-shape timings,
memory results, and an explanation of the collapse-preserving kernels.

Run from the repository root with the local PyTorch environment. Both scripts
support `--op softmax`, `--op log_softmax`, or `--op both` (the default), and
`--dtype float32` (default), `float16`, or `bfloat16`. The 20 original model
shapes are shared in `softmax_common.py`; `--cases 0 4` selects two small cases.
Only one shape's input and output are resident at a time, including the largest
shape's approximately 9.3 GiB of FP32 input and output on a 24 GB machine.

```sh
./venv/bin/python benchmarks/mps/log_softmax.py --output agent_space/latency.json
./venv/bin/python benchmarks/mps/log_softmax_memory.py --op softmax --output agent_space/softmax_memory.json
./venv/bin/python benchmarks/mps/log_softmax_memory.py --op log_softmax --output agent_space/log_softmax_memory.json
```

Latency uses preallocated output buffers, five warmup calls per operator, and
eight trials. A pilot chooses 1-300 calls per trial targeting 200 ms; `--steps`
overrides that choice. Operators alternate order between trials. The first
trial is excluded from summaries. MPS event intervals include queue gaps;
the separately recorded wall times include batch submission and synchronization.
Neither measurement isolates a single shader or includes output allocation.

Memory uses allocating operator calls. It records allocator peaks while the
input is live, live driver memory, and snapshots after tensor deletion and
cache cleanup. Driver snapshots do not measure peak driver usage. The model
shape suite is followed by the original two dynamic-shape sweeps and large
probe; `--suite models` or `--suite sweep` selects one. Run each operator in a
fresh process when comparing retained memory, since backend resources persist
across operators in the same process.

Build both revisions with the same configuration using
`pip install -e . -v --no-build-isolation`. Use the same saved benchmark source,
arguments, machine, and idle conditions for both revisions. JSON results include
the imported build revision, checkout revision, kernel source hash, options,
and all measurements. The acceptance limit for this comparison is at most 5%
latency regression per shape for each operator. Repeat results near the noise
range and inspect individual shapes rather than relying on an overall mean.

`compare_softmax.py` checks all 40 operator/shape combinations and returns a
nonzero exit code if any median exceeds the limit. Pass all repeated runs of
the same kernel to pool their measured trials; it rejects missing shapes,
mixed dtypes, and mixed kernel versions within a group.

```sh
./venv/bin/python benchmarks/mps/compare_softmax.py \
  --baseline agent_space/main_latency.json agent_space/main_latency_repeat.json \
  --candidate agent_space/candidate_latency.json agent_space/candidate_latency_repeat.json
```

The optional model correctness check copies three complete reduction rows
per shape to CPU float64, avoiding a second full-size copy of the largest
tensors. It covers both operators and all three supported dtypes. The broader
numerical and collapse regressions are in `test/test_mps.py`.

```sh
./venv/bin/python benchmarks/mps/test_softmax.py
./venv/bin/python test/test_mps.py -k softmax
```
