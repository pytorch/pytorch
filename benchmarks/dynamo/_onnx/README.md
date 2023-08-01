This folder contains python scripts to extend the dynamo benchmarking framework for benchmarking ONNX export. Bash scripts to run the benchmark and generate markdown reports are also provided.

# Usage

## Setup

It is recommended to create a fresh python environment, clone and build PyTorch from source.

```bash
# NOTE: It is required to build PyTorch with `develop`.
# It is highly recommended to build PyTorch with CUDA support.
USE_CUDA=1 python setup.py develop
```

Run the following script to install the benchmark dependencies:

```bash
./0_build_bench.sh
```

## Run Benchmark and Generate Report

```bash
./1_bench_and_report.sh
```

## Bench single model

```bash
./bench.sh --filter BertForMaskedLM
```
