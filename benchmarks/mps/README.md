# MPS softmax benchmarks

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

These scripts measure performance. Numerical and collapse regression tests
are in `test/test_mps.py` and should pass separately before accepting a kernel.
