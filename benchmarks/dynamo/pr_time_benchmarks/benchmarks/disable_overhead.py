import sys

from benchmark_base import BenchmarkBase

import torch


class Benchmark(BenchmarkBase):
    # torch._dynamo.disable is used on hot paths (e.g. around custom operators),
    # so the per-call overhead of its wrapper -- toggling the eval-frame handler
    # off and back on around the call -- matters. Measure it with a trivial
    # callee so the count reflects the wrapper, not the callee body.
    def __init__(self):
        super().__init__(
            category="disable_overhead",
            device="cpu",
        )

    def name(self):
        return self.category()

    def description(self):
        return "per-call overhead of torch._dynamo.disable with a trivial callee"

    def _prepare_once(self):
        torch._dynamo.reset()

        def f(x):
            return x

        self._fn = torch._dynamo.disable(f)
        self.a = torch.ones(1)

        # warm up
        for _ in range(10):
            self._work()

    def _prepare(self):
        pass

    def _work(self):
        self._fn(self.a)


class BenchmarkCallWithDisable(BenchmarkBase):
    # The C-level fast path: torch._C._dynamo.eval_frame.call_with_disable
    # toggles the eval-frame handler and forwards args via vectorcall in C,
    # avoiding the Python wrapper layers of torch._dynamo.disable. Compare this
    # count against disable_overhead above.
    def __init__(self):
        super().__init__(
            category="call_with_disable_overhead",
            device="cpu",
        )

    def name(self):
        return self.category()

    def description(self):
        return "per-call overhead of torch._C._dynamo.eval_frame.call_with_disable"

    def _prepare_once(self):
        from torch._C._dynamo.eval_frame import call_with_disable

        torch._dynamo.reset()

        def f(x):
            return x

        self._call_with_disable = call_with_disable
        self._f = f
        self.a = torch.ones(1)

        # warm up
        for _ in range(10):
            self._work()

    def _prepare(self):
        pass

    def _work(self):
        self._call_with_disable(self._f, self.a)


class BenchmarkDisableDynamo(BenchmarkBase):
    # torch._compile._disable_dynamo is the torch-internal disable used on hot
    # paths like torch.library.custom_op and torch.optim. It routes fully
    # recursive, non-export calls through call_with_disable.
    def __init__(self):
        super().__init__(
            category="disable_dynamo_overhead",
            device="cpu",
        )

    def name(self):
        return self.category()

    def description(self):
        return "per-call overhead of torch._compile._disable_dynamo"

    def _prepare_once(self):
        from torch._compile import _disable_dynamo

        torch._dynamo.reset()

        @_disable_dynamo
        def f(x):
            return x

        self._fn = f
        self.a = torch.ones(1)

        # warm up (also resolves the cached fast-path hooks)
        for _ in range(10):
            self._work()

    def _prepare(self):
        pass

    def _work(self):
        self._fn(self.a)


def main():
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)
    BenchmarkCallWithDisable().enable_instruction_count().collect_all().append_results(
        result_path
    )
    BenchmarkDisableDynamo().enable_instruction_count().collect_all().append_results(
        result_path
    )


if __name__ == "__main__":
    main()
