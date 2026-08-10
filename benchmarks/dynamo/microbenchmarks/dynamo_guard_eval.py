import time
import timeit

import numpy as np

import torch
import torch._dynamo.config
from torch._dynamo.eval_frame import _debug_get_cache_entry_list


# to satisfy linter complaining about undefined variable
foo = None

args = [f"x{i}" for i in range(100)]
fn_str = f"""\
def foo({", ".join(args)}):
    n = {" + ".join(arg + ".shape[0]" for arg in args)}
    return x0 + n
"""

exec(fn_str, globals())
torch._dynamo.config.recompile_limit = 16


def bench(name, fn):
    torch._dynamo.reset()
    inps = [[torch.randn(i) for _ in range(100)] for i in range(10, 101, 10)]

    def run_fn():
        for inp in inps:
            fn(*inp)

    start = time.perf_counter()
    for _ in range(3):
        run_fn()
    end = time.perf_counter()

    results = timeit.repeat(lambda: run_fn(), number=1000, repeat=10)
    print(f"{name} {np.median(results) * 1000:.1f}us (warmup={end - start:.1f}s)")


# Models with many repeated submodules (e.g. decoder stacks) exercise guard
# evaluation that scales with the number of inlined nn.Modules: per-layer scalar
# attribute guards, object-aliasing guards from shared objects, and relational
# symbolic-shape guards. See https://github.com/pytorch/pytorch/issues/185886.
#
# The same module pattern is mirrored in
# benchmarks/dynamo/pr_time_benchmarks/benchmarks/nn_module_guard_eval.py; keep
# the two in sync.
class _Shared:
    def __init__(self):
        self.flag = True


class _RepeatedLayer(torch.nn.Module):
    def __init__(self, idx, hidden, shared):
        super().__init__()
        self.eps = 1e-6
        self.scale = 0.5 + idx * 1e-3
        self.register_buffer("bias", torch.randn(hidden))
        self.shared = shared

    def forward(self, x):
        # Load-bearing: reading the shared object is what makes Dynamo emit the
        # object-aliasing guards (layers[i].shared is layers[j].shared) this
        # benchmark exercises. `flag` is always True, so do not "simplify" the
        # branch away -- doing so silently removes those guards.
        if self.shared.flag:
            x = x + self.eps
        return x * self.scale + self.bias


class _RepeatedModules(torch.nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        shared = _Shared()
        self.layers = torch.nn.ModuleList(
            [_RepeatedLayer(i, hidden, shared) for i in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def bench_guard_check(name, n_layers, hidden=64):
    # Time the compiled guard tree in isolation (guard_manager.check) so the
    # measurement reflects guard-eval cost, not the trivial per-layer compute.
    torch._dynamo.reset()
    model = _RepeatedModules(n_layers, hidden)
    compiled = torch.compile(model, fullgraph=True, dynamic=True)
    x = torch.randn(128, hidden)[:64]  # a view, so it has a ._base

    start = time.perf_counter()
    compiled(x)
    end = time.perf_counter()

    guard_manager = _debug_get_cache_entry_list(type(model).forward.__code__)[
        0
    ].guard_manager
    f_locals = {"self": model, "x": x}
    if not guard_manager.check(f_locals):
        raise RuntimeError("guard check failed; cannot benchmark guard eval")

    results = timeit.repeat(
        lambda: guard_manager.check(f_locals), number=1000, repeat=20
    )
    print(f"{name} {np.median(results) * 1000:.1f}us (warmup={end - start:.1f}s)")


def main():
    bench("compiled", torch.compile(foo, dynamic=False))  # type: ignore[F821]
    for n_layers in (10, 40, 80):
        bench_guard_check(f"repeated_modules[{n_layers}]", n_layers)


if __name__ == "__main__":
    main()
