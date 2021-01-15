# Instruction count microbenchmarks
## Quick start: A/B testing

A more detailed description is provided in later sections, however if you are
just interested in testing changes then this section should be sufficient.

#### Source command

The benchmark suite is designed to be workflow agnostic. It takes a command
to specify an environment, and as long as `import torch` works the benchmark
should as well. For instance, in a git-worktree based workflow the source
command would be something like
`cd SOME_PYTORCH_ROOT && source actvate ENV_FOR_THIS_ROOT`, while in an
"install in separate conda envs" workflow the command would be
`source activate SOME_CONDA_ENV`.

#### Testing older PyTorch versions

For older PyTorch versions, certain features which the benchmark expects may
be missing. `Timer` provides a utility to back-port itself to older versions,
which the benchmark can be told to use. Note that this back-port is (mostly)
non-destructive; it replicates to a `utils_backport` directory rather than
trampling the existing torch utils. For A/B testing, this functionality is
exposed with the `--patch_a` and `--patch_b` for the A and B environments
respectively.

Furthermore, for very old versions certain benchmarks may simply not work.
Breaking changes to TorchScript might prevent old versions from loading
models, certain APIs (e.g. `torch::functional`) might not be present, etc.
The `--backtesting` tells the benchmark to skip cases which are known not to
work on very old versions. For backtesting prior to PyTorch 1.0, breaking
changes in the header structure prevent Timer from running C++ snippets, and
the `--no_cpp` should be passed to only run Python benchmarks.

#### Example command

Suppose I have two conda environments: `fbcode_warm` and `my_awesome_branch`.
The command to measure them across a range of tasks spanning the core PyTorch
runtime would be:

```
# From PyTorch root
$ cd benchmarks/instruction_counts
$ python main.py --mode A/B \
    --A "source activate fbcode_warm" \
    --B "source activate my_awesome_branch"
```

Now suppose I want to know how my branch compares to PyTorch 1.0, which I've
installed in the branch `backtest_1_0`:

```
# From PyTorch root
$ cd benchmarks/instruction_counts
$ python main.py --mode A/B \
    --A "source activate backtest_1_0" \
    --patch_a \
    --B "source activate my_awesome_branch" \
    --backtest
```

## TODO: Write full technical overview.
