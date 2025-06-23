# PyTorch Data Benchmarks

This directory contains benchmarks for the `torch.utils.data` module components, including:

- Samplers (SequentialSampler, RandomSampler, BatchSampler, etc.)
- DataLoader
- Dataset implementations

These benchmarks help measure and track the performance of data loading and sampling operations in PyTorch.

## Dependencies

The benchmarks require the following dependencies:
```
numpy
tabulate
```

You can install them using pip:
```bash
pip install numpy tabulate
```

## Running the benchmarks

To run benchmarks for all sampler types:
```bash
python -m benchmarks.data.run_all
```

You can specify which samplers to benchmark:
```bash
python -m benchmarks.data.run_all --samplers BatchSampler,RandomSampler
```

Control the number of benchmark iterations:
```bash
python -m benchmarks.data.run_all --avg-times 20
```

Limit the number of parameter combinations tested:
```bash
python -m benchmarks.data.run_all --max-combinations 5
```

## Sampler Benchmarks

The `samplers_bench.py` script provides a unified benchmarking tool for all PyTorch sampler implementations. It can benchmark both built-in samplers and custom implementations, comparing their performance against alternative implementations.

### Basic Usage

To benchmark the default samplers:
```bash
python -m benchmarks.data.samplers_bench
```

This will benchmark BatchSampler, RandomSampler, and SequentialSampler with default parameters.

### Advanced Usage

Specify which samplers to benchmark:
```bash
python -m benchmarks.data.samplers_bench --samplers BatchSampler,WeightedRandomSampler
```

Control the number of benchmark iterations:
```bash
python -m benchmarks.data.samplers_bench --avg-times 20
```

Limit the number of parameter combinations tested:
```bash
python -m benchmarks.data.samplers_bench --max-combinations 5
```

### Custom Samplers

You can benchmark custom sampler implementations by providing a module path:
```bash
python -m benchmarks.data.samplers_bench --custom-module my_project.custom_samplers --samplers MyCustomSampler
```

### Adding Alternative Implementations

To benchmark alternative implementations against the original ones:

1. Add your alternative implementation class to `samplers_bench.py`
2. Register it in the `ALTERNATIVE_IMPLEMENTATIONS` dictionary
3. Run the benchmark to compare performance

Example:
```python
# In samplers_bench.py
class MyImprovedBatchSampler(Sampler[List[int]]):
    # Your optimized implementation here
    ...

# Register your implementation
ALTERNATIVE_IMPLEMENTATIONS = {
    BatchSampler: MyImprovedBatchSampler,
    # Other mappings...
}
```
