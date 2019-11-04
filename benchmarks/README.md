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

Run a benchmark on the branch heads configured in `asv.conf.json`::

    asv run

Right now only `master` is included, although you can add other branches
if you would like to compare those. Pass `-v` if you would like to monitor
the build progress.

Note that this will build an isolated conda environment to run the
benchmarks in and might take some time, use `--python=same` to use the
python version that `asv` is installed with. This will be much faster if
pytorch is already installed in that environment.

You can also pass `asv run` the `--skip-existing-commits` argument to
skip commits where the benchmarks have already run and there are results
present in the `asv` results database.

To compare change in benchmark results across a set of revisions

    asv run first_revision..second_revision

Where first_revision is a ref pointing to the first revision to test and
second_revision is a ref pointing to the second revision. You can pass
`--steps` to limit the number of commits in between that get benchmarked.

To generate and view the benchmark results locally, do::

    asv publish
    asv preview

You can also do::

    asv compare rev1 rev2

Where rev1 and rev2 are commit refs or hashes of commits that are
present in the results database.

More on how to use ``asv`` can be found in the [ASV
documentation](https://asv.readthedocs.io).  Command-line help is
available as usual via `asv --help` and `asv run --help`.

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
