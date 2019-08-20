# PyTorch Benchmarks

NOTE: This folder is currently work in progress.

This folder contains scripts that produce reproducible timings of various PyTorch features.

It also provides mechanisms to compare PyTorch with other frameworks.

## Setup environment
Make sure you're on a machine with CUDA, torchvision, and pytorch installed. Install in the following order:
```
# Install torchvision. It comes with the pytorch stable release binary
conda install pytorch torchvision -c pytorch

# Install the latest pytorch master from source.
# It should supercede the installation from the release binary.
cd $PYTORCH_HOME
python setup.py build develop

# Check the pytorch installation version
python -c "import torch; print(torch.__version__)"
```

## Benchmark List

Please refer to each subfolder to discover each benchmark suite

* [Fast RNNs benchmarks](fastrnns/README.md)



## PyTorch ASV benchmarks

Benchmarking PyTorch with Airspeed Velocity.


Usage
-----

Airspeed Velocity manages building and Python virtualenvs or conda envs by
itself, unless told otherwise (e.g. with `--python=same`).
To run the benchmarks, you do not need to install a development version of
PyTorch to your current Python environment.
TODO: check that the isolated build feature works, so far just used
`--python=same`.

Run a benchmark against currently installed PyTorch version (don't
record the result)::

    asv run --python=same

Compare change in benchmark results to another version::

    TODO

Run ASV commands (record results and generate HTML)::

    cd benchmarks
    asv run --skip-existing-commits --steps 10 ALL
    asv publish
    asv preview

More on how to use ``asv`` can be found in the
[ASV documentation](https://asv.readthedocs.io).
Command-line help is available as usual via `asv --help` and `asv run --help`.



Writing benchmarks
------------------

See the ASV documentation for basics on how to write benchmarks.

Some things to consider:

- The benchmark suite should be importable with any PyTorch version.

- The benchmark parameters etc. should not depend on which PyTorch version
  is installed.

- Try to keep the runtime of the benchmark reasonable.

- Prefer ASV's `time_` methods for benchmarking times rather than cooking up
  time measurements via `time.clock`, even if it requires some juggling when
  writing the benchmark.

- Preparing input tensors etc. should generally be put in the `setup` method
  rather than the `time_` methods, to avoid counting preparation time together
  with the time of the benchmarked operation.
