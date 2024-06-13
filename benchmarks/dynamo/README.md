# `torch.compile()` Benchmarking

This directory contains benchmarking code for TorchDynamo and many
backends including TorchInductor.  It includes three main benchmark suites:

- [TorchBenchmark](https://github.com/pytorch/benchmark): A diverse set of models, initially seeded from
highly cited research models as ranked by [Papers With Code](https://paperswithcode.com).  See [torchbench
installation](https://github.com/pytorch/benchmark#installation) and `torchbench.py` for the low-level runner.
[Makefile](Makefile) also contains the commands needed to setup TorchBenchmark to match the versions used in
PyTorch CI.

- Models from [HuggingFace](https://github.com/huggingface/transformers): Primarily transformer models, with
representative models chosen for each category available.  The low-level runner (`huggingface.py`) automatically
downloads and installs the needed dependencies on first run.

- Models from [TIMM](https://github.com/huggingface/pytorch-image-models): Primarily vision models, with representative
models chosen for each category available.  The low-level runner (`timm_models.py`) automatically downloads and
installs the needed dependencies on first run.


## GPU Performance Dashboard

Daily results from the benchmarks here are available in the [TorchInductor
Performance Dashboard](https://hud.pytorch.org/benchmark/compilers),
currently run on an NVIDIA A100 GPU.

The [inductor-perf-test-nightly.yml](https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml)
workflow generates the data in the performance dashboard.  If you have the needed permissions, you can benchmark
your own branch on the PyTorch GitHub repo by:
1) Select "Run workflow" in the top right of the [workflow](https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml)
2) Select your branch you want to benchmark
3) Choose the options (such as training vs inference)
4) Click "Run workflow"
5) Wait for the job to complete (4 to 12 hours depending on backlog)
6) Go to the [dashboard](https://hud.pytorch.org/benchmark/compilers)
7) Select your branch and commit at the top of the dashboard

The dashboard compares two commits a "Base Commit" and a "New Commit".
An entry such as `2.38x → 2.41x` means that the performance improved
from `2.38x` in the base to `2.41x` in the new commit.  All performance
results are normalized to eager mode PyTorch (`1x`), and higher is better.


## CPU Performance Dashboard

The [TorchInductor CPU Performance
Dashboard](https://github.com/pytorch/pytorch/issues/93531) is tracked
on a GitHub issue and updated periodically.

## Running Locally

Raw commands used to generate the data for
the performance dashboards can be found
[here](https://github.com/pytorch/pytorch/blob/641ec2115f300a3e3b39c75f6a32ee3f64afcf30/.ci/pytorch/test.sh#L343-L418).

To summarize there are three scripts to run each set of benchmarks:
- `./benchmarks/dynamo/torchbench.py ...`
- `./benchmarks/dynamo/huggingface.py ...`
- `./benchmarks/dynamo/timm_models.py ...`

Each of these scripts takes the same set of arguments.  The ones used by dashboards are:
- `--accuracy` or `--performance`: selects between checking correctness and measuring speedup (both are run for dashboard).
- `--training` or `--inference`: selects between measuring training or inference (both are run for dashboard).
- `--device=cuda` or `--device=cpu`: selects device to measure.
- `--amp`, `--bfloat16`, `--float16`, `--float32`:  selects precision to use `--amp` is used for training and `--bfloat16` for inference.
- `--cold-start-latency`: disables caching to accurately measure compile times.
- `--backend=inductor`: selects TorchInductor as the compiler backend to measure.  Many more are available, see `--help`.
- `--output=<filename>.csv`: where to write results to.
- `--dynamic-shapes --dynamic-batch-only`: used when the `dynamic` config is enabled.
- `--disable-cudagraphs`: used by configurations without cudagraphs enabled (default).
- `--freezing`: enable additional inference-only optimizations.
- `--cpp-wrapper`: enable C++ wrapper code to lower overheads.
- `TORCHINDUCTOR_MAX_AUTOTUNE=1` (environment variable): used to measure max-autotune mode, which is run weekly due to longer compile times.
- `--export-aot-inductor`: benchmarks ahead-of-time compilation mode.
- `--total-partitions` and `--partition-id`: used to parallel benchmarking across different machines.

For debugging you can run just a single benchmark by adding the `--only=<NAME>` flag.

A complete list of options can be seen by running each of the runners with the `--help` flag.

As an example, the commands to run first line of the dashboard (performance only) would be:
```
./benchmarks/dynamo/torchbench.py --performance --training --amp --backend=inductor --output=torchbench_training.csv
./benchmarks/dynamo/torchbench.py --performance --inference --bfloat16 --backend=inductor --output=torchbench_inference.csv

./benchmarks/dynamo/huggingface.py --performance --training --amp --backend=inductor --output=huggingface_training.csv
./benchmarks/dynamo/huggingface.py --performance --inference --bfloat16 --backend=inductor --output=huggingface_inference.csv

./benchmarks/dynamo/timm_models.py --performance --training --amp --backend=inductor --output=timm_models_training.csv
./benchmarks/dynamo/timm_models.py --performance --inference --bfloat16 --backend=inductor --output=timm_models_inference.csv
```
