# Benchmarking tool for the autograd API

This folder contain a set of self-contained scripts that allow to benchmark the autograd with different common models.
It is designed to run the benchmark before and after your change and will generate a table to share on the PR.

To do so, you can use `functional_autograd_benchmark.py` to run the benchmarks before your change (using as output `before.txt`) and after your change (using as output `after.txt`).
You can then use `compare.py` to get a markdown table comparing the two runs.

The default arguments of `functional_autograd_benchmark.py` should be used in general. You can change them though to force a given device to run the benchmark on or force running even the (very) slow settings.
