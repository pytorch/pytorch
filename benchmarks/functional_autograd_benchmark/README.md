# Benchmarking tool for the autograd API

This folder contain a set of self-contained scripts that allow to benchmark the autograd with different common models.
It is designed to run the benchmark before and after your change and will generate a table to share on the PR.

To do so, you can use `functional_autograd_benchmark.py` to run the benchmarks before your change (using as output `before.txt`) and after your change (using as output `after.txt`).
You can then use `compare.py` to get a markdown table comparing the two runs.

The default arguments of `functional_autograd_benchmark.py` should be used in general. You can change them though to force a given device or force running even the (very) slow settings.


### Files in this folder:
- `functional_autograd_benchmark.py` is the main entry point to run the benchmark.
- `compare.py` is the entry point to run the comparison script that generates a markdown table.
- `torchaudio_models.py` and `torchvision_models.py`  contains code extracted from torchaudio and torchvision to be able to run the models without having a specific version of these libraries installed.
- `ppl_models.py`, `vision_models.py` and `audio_text_models.py` contain all the getter functions used for the benchmark.
