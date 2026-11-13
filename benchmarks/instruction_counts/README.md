# Instruction count microbenchmarks
## Quick start

### To run the benchmark:

```
# From pytorch root
cd benchmarks/instruction_counts
python main.py
```

Currently `main.py` contains a very simple threadpool (so that run time isn't
unbearably onerous) and simply prints the results. These components will be
upgraded in subsequent PRs.

### To define a new benchmark:
* `TimerArgs`: Low level definition which maps directly to
`torch.utils.benchmark.Timer`
* `GroupedStmts`: Benchmark a snippet (Python, C++, or both). Can automatically
generate autograd variants.
* `GroupedModules`: Like `GroupedStmts`, but takes `nn.Module`s
* `GroupedVariants`: Benchmark-per-line to define many related benchmarks in a
single code block.

## Architecture
### Benchmark definition.

One primary goal of this suite is to make it easy to define semantically
related clusters of benchmarks. The crux of this effort is the
`GroupedBenchmark` class, which is defined in `core/api.py`. It takes a
definition for a set of related benchmarks, and produces one or more concrete
cases. It's helpful to see an example to understand how the machinery works.
Consider the following benchmark:

```
# `GroupedStmts` is an alias of `GroupedBenchmark.init_from_stmts`
benchmark = GroupedStmts(
    py_stmt=r"y = x * w",
    cpp_stmt=r"auto y = x * w;",

    setup=GroupedSetup(
        py_setup="""
            x = torch.ones((4, 4))
            w = torch.ones((4, 4), requires_grad=True)
        """,
        cpp_setup="""
            auto x = torch::ones((4, 4));
            auto w = torch::ones((4, 4));
            w.set_requires_grad(true);
        """,
    ),

    signature="f(x, w) -> y",
    autograd=True,
),
```

It is trivial to generate Timers for the eager forward mode case (ignoring
`num_threads` for now):

```
Timer(
    stmt=benchmark.py_fwd_stmt,
    setup=benchmark.setup.py_setup,
)

Timer(
    stmt=benchmark.cpp_fwd_stmt,
    setup=benchmark.setup.cpp_setup,
    language="cpp",
)
```

Because we know the output variable (`y`), generating an autograd variant is
as simple as appending `y.backward()` (or `y.backward();` in C++) to the stmt
of the forward only variant. This requires that `signature` be provided, as
there is nothing special about the name `y`.

The logic for the manipulations above is split between `core/api.py` (for
generating model invocations based on language) and `core/expand.py` (for
expanding forward, autograd, language, and thread-count variants). The benchmarks
themselves are defined in `definitions/standard.py`. The current set is chosen
to demonstrate the various model definition APIs, and will be expanded when the
benchmark runner infrastructure is better equipped to deal with a larger run.

### Benchmark execution.

Once `expand.materialize` has flattened the abstract benchmark definitions into
`TimerArgs`, they can be sent to a worker (`worker/main.py`) subprocess to
execution. This worker has no concept of the larger benchmark suite; `TimerArgs`
is a one-to-one and direct mapping to the `torch.utils.benchmark.Timer` instance
that the worker instantiates.
