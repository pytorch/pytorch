# Instruction count microbenchmarks
## __Quick start: A/B testing__

A more detailed description is provided in later sections, however if you are
just interested in testing changes then this section should be sufficient.

#### Source command

The benchmark suite is designed to be workflow agnostic. It takes a command
to specify an environment, and as long as `import torch` works the benchmark
should as well. For instance, in a git-worktree based workflow the source
command would be something like
`cd SOME_PYTORCH_ROOT && source activate ENV_FOR_THIS_ROOT`, while in an
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

## Quick start: Ad-hoc A/B testing

The default benchmark suite is reasonably comprehensive, but as a result it is
also somewhat slow and cannot cover any one area of the code base in great
detail. As a result, it is not well suited to A/B testing narrow PRs, such as
those modifying kernel implementations. For this reason, a second benchmark
definition is provided in `definitions/ad_hoc.py`. (Along with some inline
quick start examples.) A/B testing can be told to use this benchmark set
instead of the standard one by passing the `--ad_hoc` flag.

To show how this works, suppose we have optimized `torch.cumsum` and want to
check if out optimizations might adversely add overhead to various workflows.
If we modify the contents of `definitions/ad_hoc.py` from:

```
ADHOC_BENCHMARKS: FlatIntermediateDefinition = flatten({
    ...
})
```

to:

```
AD_HOC_SETUP = GroupedSetup(
    py_setup=r"""
        # Blocks, multi-line definitions, and comments are all fine.
        x_2d = torch.ones((2, 2))
    """,
    cpp_setup=r"""
        // Blocks, multi-line definitions, and comments are all fine.
        auto x_2d = torch::ones({2, 2});
    """
)


ADHOC_BENCHMARKS: FlatIntermediateDefinition = flatten({
    "torch.cumsum (2D)": {
        "dim 0": GroupedStmts(
            r"torch.cumsum(x_2d, dim=0)",
            r"torch::cumsum(x_2d, /*dim=*/0);",
            setup=AD_HOC_SETUP,
        ),

        ("dim 0", "discontiguous"): GroupedStmts(
            r"torch.cumsum(x_2d.t(), dim=0)",
            r"torch::cumsum(x_2d.t(), /*dim=*/0);",
            setup=AD_HOC_SETUP,
        ),

        "dim 1": GroupedStmts(
            r"torch.cumsum(x_2d, dim=1)",
            r"torch::cumsum(x_2d, /*dim=*/1);",
            setup=AD_HOC_SETUP,
        ),

        ("dim 1", "discontiguous"): GroupedStmts(
            r"torch.cumsum(x_2d.t(), dim=1)",
            r"torch::cumsum(x_2d.t(), /*dim=*/1);",
            setup=AD_HOC_SETUP,
        ),

        "both dims": GroupedStmts(
            r"""
                y = torch.cumsum(x_2d, dim=0)
                z = torch.cumsum(y, dim=1)
            """,
            r"""
                auto y = torch::cumsum(x_2d, /*dim=*/0);
                auto z = torch::cumsum(y, /*dim=*/1);
            """,
            setup=AD_HOC_SETUP,
            signature="f(x_2d) -> z",
            torchscript=True,
        )
    }
})
```

and then modify our earlier command line invocation:

```
# From PyTorch root
$ cd benchmarks/instruction_counts
$ python main.py --mode A/B \
    --A "source activate fbcode_warm" \
    --B "source activate my_awesome_branch" \
    --ad_hoc  # Point to our `cumsum` specific benchmark
```

we can test only on the more targeted benchmarks.

## Design

### Benchmark definition

One primary goal of this suite is to make it easy to define semantically
related clusters of benchmarks. (See the "Benchmark definition (Explicit)"
section if this is not required.) For instance, consider the snippet:

```
# Setup
x = torch.ones((4, 1))
v = torch.ones((1,), requires_grad=True)

# Stmt
y = torch.nn.functional.relu(x * v)
```

and the C++ analog:

```
// Setup
auto x = torch::ones({4, 1});
auto v = torch::ones({1});
v.set_requires_grad(true);

// Stmt
auto y = torch::nn::functional::relu(x * v);
```

If we know that creation of `x` and `v` is part of setup, and the overall
comptation uses `x` and `v` to produce `y` then we can derive TorchScript'd
and AutoGrad variants as well. The syntax for declaring that relationship
in the benchmark is `signature="f(x, y) -> y"`. For instance we can deduce that
a TorchScript model will take the form:

```
@torch.jit.script
def f(x, v):
    y = torch.nn.functional.relu(x * v)
    return y
```

and that we can add AutoGrad by appending a `y.backward()` (or `y.backward();`
in C++) line after the primary stmt. These transformations are mechanical,
however they can be quite tedious and error prone to write. (Or update!) The
result is also quite difficult to read, with the full cross product of
`{Python, C++} x {Eager, JIT} x {Forward, Forward+Backward}` comprising many
dozens of lines of mostly redundant information.

To simplify the process and keep benchmarks readable, several `Grouped` helpers
are provided which take only the minimal information needed to generate all
requested configurations. For instance in the example above, we would define:

```
example_setup = GroupedSetup(
    """
        x = torch.ones((4, 1))
        v = torch.ones((1,), requires_grad=True)
    """,
    """
        auto x = torch::ones({4, 1});
        auto v = torch::ones({1});
        v.set_requires_grad(true);
    """
)

# `flatten` is a util for internal bookkeeping.
BENCHMARKS = flatten({
    "Example": GroupedStmts(
        "y = torch.nn.functional.relu(x * v)",
        "auto y = torch::nn::functional::relu(x * v);",
        setup=example_setup,
        signature="f(x, v) -> y",
        torchscript=True,
        autograd=True,
    )
})
```

A more complete API description is available in `core/api.py`. Note that our
example has an `"Example"` tag. `core/types.py` gives a comprehensive overview
of the label scheme. (TL;DR Nested dicts with string and Tuple[str, ...] keys
are supported.)

### Benchmark definition (Explicit)

If one does not wish to use the Grouped syntax, a much simplified `TimerArgs`
is provided as well. For instance:

```
BENCHMARKS = flatten({
    "Example": TimerArgs(
        "y = torch.nn.functional.relu(x * v)",
        setup="""
            x = torch.ones((4, 1))
            v = torch.ones((1,), requires_grad=True)
        """,
        language=Language.PYTHON,
    )
})
```

This simply forwards the arguments to the constructor of
`torch.utils.benchmark.Timer`.
