import sys

from benchmark_base import BenchmarkBase

import torch
import torch.nn as nn
from torch._dynamo.eval_frame import _debug_get_cache_entry_list


# Guard evaluation for models with many repeated submodules (e.g. decoder
# stacks) scales with the number of inlined nn.Modules: per-layer scalar
# attribute guards, object-aliasing guards from shared objects, and relational
# symbolic-shape guards. This benchmark guards against regressions in that
# runtime guard-eval cost. See https://github.com/pytorch/pytorch/issues/185886.
#
# The same module pattern is mirrored in
# benchmarks/dynamo/microbenchmarks/dynamo_guard_eval.py; keep the two in sync.
class Shared:
    def __init__(self):
        self.flag = True


class GuardEvalLayer(nn.Module):
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


class RepeatedModules(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        shared = Shared()
        self.layers = nn.ModuleList(
            [GuardEvalLayer(i, hidden, shared) for i in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Benchmark(BenchmarkBase):
    N_LAYERS = 20
    HIDDEN = 64

    def __init__(self, dynamic):
        super().__init__(
            category="nn_module_guard_eval",
            backend="eager",
            device="cpu",
            dynamic=dynamic,
        )

    def name(self):
        prefix = self.category()
        if self.is_dynamic():
            prefix += "_dynamic"
        return prefix

    def description(self):
        return "runtime guard-eval cost for a stack of repeated nn.Modules"

    def _prepare_once(self):
        torch._dynamo.reset()
        self._model = RepeatedModules(self.N_LAYERS, self.HIDDEN)
        # Pin enable_cpp_symbolic_shape_guards so the dynamic baseline is robust
        # to a future default flip: with it True the symbolic-shape guards move
        # into a compiled C++ guard, shifting the instruction count with no
        # change to expected_results.csv. (The static variant has no
        # symbolic-shape guards and is unaffected.)
        with torch._dynamo.config.patch(enable_cpp_symbolic_shape_guards=False):
            compiled = torch.compile(
                self._model,
                fullgraph=True,
                backend=self.backend(),
                dynamic=self.is_dynamic(),
            )
            # a view, so the input has a ._base (exercises relational shape guards)
            self._x = torch.randn(2 * self.HIDDEN, self.HIDDEN)[: self.HIDDEN]
            compiled(self._x)

        self._guard_manager = _debug_get_cache_entry_list(
            type(self._model).forward.__code__
        )[0].guard_manager
        self._f_locals = {"self": self._model, "x": self._x}
        if not self._guard_manager.check(self._f_locals):
            raise RuntimeError("guard check failed; cannot benchmark guard eval")

    def _prepare(self):
        pass

    def _work(self):
        # Time guard evaluation in isolation so the metric reflects guard-eval
        # cost rather than the trivial per-layer compute.
        self._guard_manager.check(self._f_locals)


def main():
    result_path = sys.argv[1]
    for dynamic in (False, True):
        Benchmark(dynamic).enable_instruction_count().collect_all().append_results(
            result_path
        )


if __name__ == "__main__":
    main()
