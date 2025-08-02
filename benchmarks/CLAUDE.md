# PyTorch Benchmarks Guide

Performance benchmarking tools and reproducible timing scripts for PyTorch features.

## 🏗️ Directory Organization

TODO(Claude): Check the git history to see when each folder was last changed and put at the top the actively used benchmark and move below the ones that haven't been touched in a while. Also remove from the "key benchmark suites" section all benchmarks that didn't see any significant change in the past 3 months

### Core Benchmarking Suites
- **`dynamo/`** - TorchDynamo compiler benchmarks with model accuracy and performance testing
- **`fastrnns/`** - Fast RNN implementation benchmarks comparing different cell types
- **`operator_benchmark/`** - Individual operator performance benchmarks
- **`functional_autograd_benchmark/`** - JAX-style functional autograd performance tests
- **`instruction_counts/`** - CPU instruction counting and analysis tools

### Domain-Specific Benchmarks
- **`sparse/`** - Sparse tensor operations benchmarking (SpMM, SpMV, CSR)
- **`transformer/`** - Attention mechanism and transformer component benchmarks
- **`tensorexpr/`** - TensorExpr/NNC fusion benchmarks
- **`gpt_fast/`** - GPT model inference optimization benchmarks
- **`nested/`** - Nested tensor operations benchmarks

### System-Level Benchmarks
- **`distributed/`** - Distributed training benchmarks (DDP)
- **`static_runtime/`** - Static runtime performance tests
- **`profiler_benchmark/`** - PyTorch profiler overhead measurements
- **`serialization/`** - Model serialization/deserialization benchmarks
- **`framework_overhead_benchmark/`** - Framework overhead analysis

### Specialized Tools
- **`inference/`** - Inference server benchmarking tools
- **`overrides_benchmark/`** - `__torch_function__` override performance
- **`fuser/`** - JIT fuser performance analysis
- **`inductor_backends/`** - Inductor backend benchmarks (CUTLASS)
- **`pr_time_benchmarks/`** - PR impact timing benchmarks for CI

## 🚀 Running Benchmarks

### Environment Setup
```bash
# Install PyTorch from source (recommended for benchmarking)
eval $BUILD_CONFIG python setup.py develop

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Key Benchmark Suites

#### Dynamo Benchmarks
```bash
cd benchmarks/dynamo
python benchmarks.py --help
python benchmarks.py --backend=inductor --suite=torchbench_inference
```

#### Operator Benchmarks
```bash
cd benchmarks/operator_benchmark
python benchmark_runner.py --operators=add,mm --backends=eager
```

#### FastRNN Benchmarks
```bash
cd benchmarks/fastrnns
python bench.py --help
python bench.py --lstm --jit
```

### Running All Benchmarks
TODO(Claude): This is part of the dynamo subfolder, move up it up in that section
```bash
# Run comprehensive benchmark suite
cd benchmarks
./run_all.sh

# Compare with baseline
./run_delta.sh
```

## 🧪 Testing Benchmarks

### Benchmark Validation
```bash
# Test dynamo benchmarks
cd benchmarks/dynamo
python test.py

# Test operator benchmarks
cd benchmarks/operator_benchmark
python benchmark_all_test.py
```

### Performance Regression Testing
```bash
# Check for performance regressions
cd benchmarks/pr_time_benchmarks
python check_results.py
```

## 📊 Analysis Tools

### Performance Analysis
- **`summarize_perf.py`** - Summarize performance results across benchmarks
- **`parse_logs.py`** - Parse benchmark logs for analysis
- **`upload_scribe.py`** - Upload results to tracking systems

### Comparison Tools
- **`compare.sh`** - Compare benchmark results between runs
- **`compare-fastrnn-results.py`** - FastRNN specific comparisons

## 📁 Key Files

### Configuration
- `run_all.sh` - Master benchmark runner script TODO(Claude): that file doesn't exist
- `dynamo/benchmarks.py` - Main Dynamo benchmark runner
- `operator_benchmark/benchmark_runner.py` - Operator benchmark runner

### Results and Data
- `dynamo/expected_ci_perf_*.csv` - Expected CI performance baselines
- `dynamo/ci_expected_accuracy/` - Model accuracy baselines
- `pr_time_benchmarks/expected_results.csv` - PR benchmark baselines

## 🐛 Common Issues

### CUDA Issues
- Ensure CUDA toolkit matches PyTorch CUDA version
- Set `CUDA_VISIBLE_DEVICES` for specific GPU testing
- Check memory usage with large models

### Benchmark Failures
- **Timeout errors**: Increase timeout limits for slow models
- **OOM errors**: Reduce batch sizes or use CPU-only mode
- **Accuracy failures**: Check model configurations and data

### Environment Issues
- **Missing dependencies**: Install benchmark-specific requirements
- **Version mismatches**: Ensure consistent PyTorch/torchvision versions

## 📝 Notes for Claude

This directory provides:
- **Comprehensive benchmarking**: Performance testing across all PyTorch components
- **CI integration**: Automated performance regression detection
- **Framework overhead measurement**: Check how expensive no-ops are
- **Model coverage**: TorchBench, HuggingFace, TIMM model suites
- **Hardware profiling**: CPU, CUDA, instruction counting
- **Compilation testing**: Dynamo, Inductor, TorchScript benchmarks

Key benchmarking practices:
- Use consistent environments and warmup runs
- Measure both forward and backward pass performance
- Include memory usage and compilation time
- Test across different batch sizes and input shapes
- Validate numerical accuracy alongside performance