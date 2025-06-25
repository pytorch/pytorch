# PyTorch Data Benchmarks

This directory contains benchmarks for the `torch.utils.data` module components, focusing on the performance of samplers.

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

To run the BatchSampler benchmark:
```bash
python -m benchmarks.data.samplers_bench
```

## BatchSampler Benchmark

The `samplers_bench.py` script benchmarks the performance of PyTorch's BatchSampler against an alternative implementation. It tests with the following parameters:

- Batch sizes: 4, 8, 64, 640, 6400, 64000
- Drop last options: True, False
- Each configuration is run 10 times and averaged
- Results include speedup percentage calculations

### Implementation Details

The benchmark compares two implementations:

1. **Original BatchSampler**: The standard PyTorch implementation
2. **Alternative BatchSampler**: An implementation with a different approach to the `__iter__` method

The alternative implementation uses two different strategies based on the `drop_last` parameter:
- When `drop_last=True`: Uses a try/except approach with StopIteration
- When `drop_last=False`: Pre-allocates batches and fills them incrementally

### Output

The benchmark outputs a table with the following columns:
- Batch Size
- Drop Last
- Original (s): Time taken by the original implementation
- New (s): Time taken by the alternative implementation
- Speedup: Percentage improvement of the new implementation over the original

Example output:
```
+------------+-----------+---------------+----------+---------+
| Batch Size | Drop Last | Original (s)  | New (s)  | Speedup |
+============+===========+===============+==========+=========+
|          4 | True      | 0.1234        | 0.1000   | 18.96%  |
+------------+-----------+---------------+----------+---------+
|          4 | False     | 0.1345        | 0.1100   | 18.22%  |
+------------+-----------+---------------+----------+---------+
...
```

### Extending the Benchmark

To benchmark a different implementation:

1. Modify the `NewBatchSampler` class in `samplers_bench.py` with your implementation
2. Run the benchmark to compare its performance against the original
