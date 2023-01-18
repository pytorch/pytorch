# Torchdynamo Benchmarks

## What We Benchmark
TorchDynamo provides a benchmark harness that takes care of uniformly benchmarking different models.  It interleaves runs of eager and dynamo to avoid machine noise/variability issues, and reports results based on medians along with P-values.

The runner integrates with models from TorchBenchmark, HuggingFace and TIMM suites and covers both training and inference.

The infrastructure allows us to specify a loss function. For torchbench models, we use .sum().backward() call in place of the native loss function. For TIMM models, we use a CrossEntropy loss. And HF models contain a loss function inside the model itself, so we don't need any special loss computation handling.

Training benchmarks approximate training by running the model forward, computing loss, running backward, and then the optimizer (SGD). Note: the optimizer is currently not compiled by Torchdynamo.

Inference benchmarks and Training benchmarks measure correctness by comparing dynamo and eager model outputs given fixed inputs and seeds.

## Setup

### Machine
We run benchmarks on AWS machines (p4d.24xlarge) using 8xNVidia A100 40GB cards.  We suggest using Cuda 11.6 for consistency.

### Benchmarks
Make sure to carefully follow the [torchbench installation](https://github.com/pytorch/benchmark#installation) instructions, taking care to build the auxiliary libraries (torchvision, torchtext) from a matching version to your pytorch version.

For HF and TIMM models, the scripts already install the transformers and timm package respectively on the first run.

## Runbook

### Basic Usage
There are a lot of flags in the benchmark runner, and it can be confusing to know which settings to use or what machine to run it on.  In order to support apples-to-apples comparison, we have provided the following 'standard' settings in `runner.py`. This script is a wrapper over the common benchmarking infrastructure and simplifies the flags. We will continually update `runner.py` with the latest and most relevant compilers for training and inference. It also provides some graph utilities to visualize and compare results. Some of the example commands are

**Inference Commands**
* Inference compilers on torchbench models - `python benchmarks/dynamo/runner.py --suites=torchbench --inference --dtypes=float16`
* Inductor Inference compiler on torchbench models - `python benchmarks/dynamo/runner.py --suites=torchbench --inference --dtypes=float16 --compilers=inductor`

**Training Commands**
* Training compilers on TIMM models - `python benchmarks/dynamo/runner.py --suites=timm_models --training --dtypes=float32 --output-dir=timm_logs`
* AOTAutograd Training compiler on TIMM models - `python benchmarks/dynamo/runner.py --suites=timm_models --training --dtypes=float32 --compilers=aot_nvfuser --output-dir=timm_logs`
* Inductor Training compiler on TIMM models - `python benchmarks/dynamo/runner.py --suites=timm_models --training --dtypes=float32 --compilers=inductor --output-dir=timm_logs`

Running runner.py generates a file named `run.sh`. This file contains the actual commands that invoke the common benchmarking infrastructure with the appropriate flags. Which brings us to the advanced usage.

### Advanced Usage

One could directly call `torchbench.py`, `huggingface.py` or `timm_models.py` with the necessary flags. There are a lot of flags in the benchmarks runner. Some of the examples are as follows. These are subject to change.

**Inference Commands**
* TorchScript (with TorchDynamo capture) NVFuser Inference - `python benchmarks/dynamo/torchbench.py -dcuda -n100 --speedup-dynamo-ts --performance`
* TorchInductor CUDA Graphs Inference - `python benchmarks/dynamo/torchbench.py -dcuda --float32 -n50 --inductor --performance`

**Training Commands**
* Torchscript (with TorchDynamo capture) NVFuser Training - `python benchmarks/dynamo/torchbench.py --float32 -dcuda --training --nvfuser --speedup-dynamo-ts --performance`
* TorchInductor CUDA Graphs Training - `python benchmarks/dynamo/torchbench.py --float32 -dcuda --training --inductor --performance`

Above commands are for torchbench models. You can simply replace `torchbench.py` with `huggingface.py` for HF models, and `timm_model.py` for TIMM models.
