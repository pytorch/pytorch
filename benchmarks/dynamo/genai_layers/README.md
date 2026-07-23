# GenAI Kernel Benchmark

This directory contains benchmarks for the GenAI kernels. It compares pytorch eager, pytorch compiler, quack, and liger.


## Setup

Assuming pytorch is installed.

```
pip install -r requirements.txt
```

## Run

```
  python benchmark.py --list                    # List all available benchmarks
  python benchmark.py --all                     # Run all benchmarks
  python benchmark.py cross_entropy_forward     # Run specific benchmark
  python benchmark.py softmax_forward softmax_backward  # Run multiple benchmarks
```

Add `--visualize` to plot graph for the benchmark results.
