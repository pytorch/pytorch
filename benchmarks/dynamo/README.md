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

### Decoder-only generation, prefill, and decode

Registered Hugging Face text-generation models use `--hf-inference-mode=generate`
by default. Prefill is opt-in and supports `meta-llama/Llama-3.2-1B`,
`google/gemma-2-2b`, `Qwen/Qwen3-0.6B`, and `Qwen/Qwen3.5-0.8B` with fixed,
unpadded prompts:

```
./benchmarks/dynamo/huggingface.py --performance --inference --bfloat16 \
  --backend=inductor --hf-inference-mode=prefill --prompt-length=1000 \
  --batch-size=1 --only=Qwen/Qwen3-0.6B --output=prefill_performance.csv
```

Prefill measures one forward over the prompt, returning last-position logits and
populating a reusable static cache. The cache holds `prompt_length + 1999`
positions, the capacity `generate` allocates for its first forward, so attention
spans the same KV length as in generation; `prompt_length + 2000` must fit the
model's context length. Cache allocation and reset occur outside the compiled
and timed forward. Use `--accuracy` to check correctness or
`--disable-cudagraphs` to measure without CUDA graphs. Prefill requires inference
and a backend; autocast, dynamic shapes, DDP/FSDP, and export modes are
unsupported. `--prompt-length` (default 1000) and `--batch-size` configure
the prefill workload; generation retains its existing `1 x 1000` inputs and
rejects `--prompt-length`.

CSV `abs_latency` and `eager_latency` are milliseconds for all
`--iterations-per-run` requests; `input_tokens_per_second` includes all batch
elements. Default output and profiler trace names get a `_prefill` suffix; use
separate files for different input dimensions.

Decode is opt-in and supports `meta-llama/Llama-3.2-1B`, `Qwen/Qwen3-0.6B`, and
`Qwen/Qwen3.5-0.8B`:

```
./benchmarks/dynamo/huggingface.py --performance --inference --bfloat16 \
  --backend=inductor --hf-inference-mode=decode --prompt-length=1000 \
  --decode-length=128 --batch-size=1 --only=Qwen/Qwen3-0.6B \
  --output=decode_performance.csv
```

Each decode request prefills the prompt eagerly into the static cache outside
the timed region, then times `--decode-length` (default 128, at most 1999)
single-token compiled forwards. Decode feeds fixed random tokens rather than
sampled ones and returns the last step's logits. It shares prefill's cache
capacity and flag restrictions. `gemma-2-2b` is excluded because its
sliding-window cache recompiles on every step.

CSV `abs_latency` and `eager_latency` are milliseconds for all
`--iterations-per-run` requests; `output_tokens_per_second` counts every decode
step across batch elements. Default output and profiler trace names get a
`_decode` suffix.

#### Tensor-parallel prefill and decode

`--tp-size=N` runs prefill or decode with one model and request sharded across
`N` GPUs of one node by Hugging Face tensor parallelism (`tp_plan="auto"`). Both
the eager baseline and the compiled forward run tensor-parallel, so `speedup`
compares eager TP with compiled TP at the same topology. Compare against a
single-GPU run separately to measure TP scaling.
Both phases support `meta-llama/Llama-3.2-1B`, `Qwen/Qwen3-0.6B`, and
`Qwen/Qwen3.5-0.8B`.

```
./benchmarks/dynamo/huggingface.py --performance --inference --bfloat16 \
  --backend=inductor --nopython --disable-cudagraphs --hf-inference-mode=decode \
  --tp-size=2 --prompt-length=1000 --decode-length=128 --batch-size=1 \
  --only=Qwen/Qwen3-0.6B
```

Without torchrun, the script launches one `torchrun --standalone
--nproc-per-node=N` worker group for each selected model (`--only`, or every
supported model that passes `--filter`). A failed group gets a failure row if
it did not already write a result; performance failures have zero metrics.
The driver continues through the remaining models, then exits unsuccessfully
if any group crashed, timed out, or produced missing or invalid results.
Completed accuracy results are evaluated by `check_accuracy.py` against the
expected CSV, so known accuracy failures can be recorded in the baseline.
Performance results require positive speedup and latency values.
It can also run as a torchrun worker directly, which
requires `--only` and `WORLD_SIZE == N` on one node:

```
torchrun --nproc-per-node=2 benchmarks/dynamo/huggingface.py --performance \
  --inference --bfloat16 --backend=inductor --disable-cudagraphs \
  --hf-inference-mode=prefill --tp-size=2 --only=Qwen/Qwen3-0.6B
```

Each worker uses `LOCAL_RANK` for its GPU and torchrun's rendezvous settings.
TP requires `--bfloat16`, `--float16`, or `--float32`, which is passed to
`from_pretrained`, and `N` must divide the model's attention and KV
heads (Qwen3.5-0.8B has two KV heads, so it supports at most `--tp-size=2`).
Ranks broadcast the request tokens from rank 0 and allocate their own
TP-local static caches. Eager, compiled, and accuracy-reference passes share the
loaded shards rather than deep-copying the model; each pass gets a fresh cache.
Hugging Face's gradient all-reduce backward hooks are removed after loading,
since they do nothing in inference and Dynamo cannot trace them.

Before each timed request, ranks finish its setup and meet at a barrier. Each
request counts as its slowest rank's duration, through device completion;
`abs_latency` and `eager_latency` are the median over repetitions of those
per-request maxima summed over `--iterations-per-run`. Token throughput counts
logical tokens only (`iterations * batch_size * prompt_length` or
`decode_length`), never multiplied by `N`. Rank 0 writes the CSV and JSON
results, which add `phase`, `tp_size`, `prompt_length`, and `decode_length`
columns, report the maximum peak memory over ranks in `eager_peak_mem` and
`dynamo_peak_mem`, and list each rank's value in `*_peak_mem_per_rank`. Default
output names get a `_<phase>_tp<N>` suffix; profiler traces and memory
snapshots get per-rank names.
Accuracy runs agree on comparison outcomes across ranks. Exceptions during
loading or execution, including the FP64 reference, terminate the worker
immediately so torchrun can stop peers blocked in model collectives. These
execution failures remain fatal even when an accuracy failure is expected.
The process-group timeout covers hangs that do not raise an exception.

TP requires `--disable-cudagraphs` until CUDA graphs are validated with it.
Multi-node runs, several TP groups, and other parallelism styles are
unsupported. Inductor TP requires CUDA/NCCL. On machines without an
accelerator, `-dcpu` supports TP diagnostics with `--backend=eager` or
`--backend=aot_eager`. CPU Inductor TP is rejected because concurrent Gloo
all-gathers can corrupt memory; this was reproduced without a model or
compilation and is not specific to Qwen3.5.

The periodic `inductor_huggingface_tp` CI job checks accuracy and measures
performance for both phases with `--tp-size=2` and full-graph compilation on a
multi-GPU A10G runner. Accuracy checks require exactly one result matching the
baseline for each supported model. Update both phase baselines with
`python benchmarks/dynamo/ci_expected_accuracy/update_expected.py <commit-sha>`.
Its performance CSV names end in `_tp2_<phase>`.
Use the numeric periodic run ID with `perf_cli.py summary` or `repro`; branch
discovery and `launch` still select the nightly workflows. Automatic baseline
selection requires a successful TP job on the same runner hardware. Comparisons
pair the same hardware, phase, TP degree, model, batch size, prompt length, and
decode length. Reproduction commands preserve those workload dimensions.

As an example, the commands to run first line of the dashboard (performance only) would be:
```
./benchmarks/dynamo/torchbench.py --performance --training --amp --backend=inductor --output=torchbench_training.csv
./benchmarks/dynamo/torchbench.py --performance --inference --bfloat16 --backend=inductor --output=torchbench_inference.csv

./benchmarks/dynamo/huggingface.py --performance --training --amp --backend=inductor --output=huggingface_training.csv
./benchmarks/dynamo/huggingface.py --performance --inference --bfloat16 --backend=inductor --output=huggingface_inference.csv

./benchmarks/dynamo/timm_models.py --performance --training --amp --backend=inductor --output=timm_models_training.csv
./benchmarks/dynamo/timm_models.py --performance --inference --bfloat16 --backend=inductor --output=timm_models_inference.csv
```
