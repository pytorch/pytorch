This folder contains python scripts to extend the dynamo benchmarking framework for benchmarking ONNX export. Bash scripts to run the benchmark and generate markdown reports are also provided.

# Usage

## Setup

It is recommended to create a fresh python environment, clone and build PyTorch from source.

It is expected that PyTorch is already built. Run the following script to install the benchmark dependencies:

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