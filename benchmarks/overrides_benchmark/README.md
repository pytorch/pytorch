# `__torch_function__` micro-benchmarks

This benchmark suite provides a systemic way to measure the performance of `__torch_function__` overhead.

## Getting started
### Initial Setup
Install `py-spy` by doing:

```bash
pip install py-spy
```

Note that more extensive documentation on using `py-spy` is available in `CONTRIBUTING.md`.

### Running the benchmark
Run one of the following commands in the terminal, with the working directory being `${PYTORCH_CLONE_DIR}/benchmarks/overrides_benchmark`:

```bash
# Benchmark all the cases
python bench.py

# Flame graph pertaining to each case.
py-spy record -o tensor.svg --native -- python pyspybench.py Tensor
py-spy record -o subtensor.svg --native -- python pyspybench.py SubTensor
py-spy record -o overridden.svg --native -- python pyspybench.py WithTorchFunction
py-spy record -o suboverridden.svg --native -- python pyspybench.py SubWithTorchFunction
```

Here is a brief overview of what the results should look like, if run correctly:

* Overhead for `torch` functions when run on `torch.Tensor` objects is on the order of 2 μs.
* `__torch_function__` should add zero overhead for `torch.Tensor` inputs, a small overhead for subclasses of `torch.Tensor`, and a couple of microseconds for `Tensor`-likes with `__torch_function__`.
* Changing the dispatching mechanism may result in changes that are on the order of 100 ns, which are hard to detect due to noise, but important.

## Reporting benchmark results
When modifying any of the machinery around `__torch_function__`, run the benchmark for both the feature branch and the point it diverges from `master`. For each of these:

* Run `bench.py`, and include the output in your result.
* For each case where `bench.py` shows a regression, run the commands described above, prefixing the output SVG filename (the input to the `-o` switch) with `base-` or `branch-` depending on the commit you are running the benchmark on.
* For each SVG, open it in the browser, take a screenshot and include it in your result. Also include a ZIP file with all SVGs thus produced included.
