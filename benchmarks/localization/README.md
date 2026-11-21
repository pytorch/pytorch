# CUDA localization benchmarks

Compare baseline and localized Linear, attention, and FFN workloads, and
benchmark localized chained Linears using bf16 and
`torch.compile(mode="reduce-overhead")`.
Requires a CUDA GPU with locality-domain support and a PyTorch build exposing
`torch.cuda.localize`.

Install the optional benchmark dependency:

```bash
python -m pip install pytest-benchmark
```

Run from the repository root:

```bash
python -m pytest benchmarks/localization/test_localization_benchmark.py --benchmark-enable
```

Use `-k simple_matmul` to select the Linear cases. For a smoke run without
repeated timing, replace `--benchmark-enable` with `--benchmark-disable`.
Missing dependencies or unsupported hardware cause the benchmark suite to skip.

These benchmarks run separately from the regular unit tests. Numerical and
compiler regression tests remain in `test/inductor/test_localization.py`.
