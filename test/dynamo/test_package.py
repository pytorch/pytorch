# Owner(s): ["module: dynamo"]

import os

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.package import _CompilePackage
from torch._inductor.runtime.runtime_utils import cache_dir


class TestPackage(torch._inductor.test_case.TestCase):
    def path(self):
        return os.path.join(cache_dir(), f"package_{self.id()}")

    def test_basic_fn(self):
        def fn(x):
            return x + 1

        args = (torch.randn(3, 2),)
        package = _CompilePackage(backend_type="eager")
        compiled_fn = torch._dynamo.optimize(backend="eager", package=package)(fn)
        expected = compiled_fn(*args)

        package.save(self.path())

        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(*args)

        package.load(self.path())
        self.assertEqual(expected, compiled_fn(*args))

    def test_graph_break_bomb(self):
        def fn(x, l, r):
            if l > r:
                return x.sum() + 1
            mid = (l + r) // 2
            if x.sum() == mid:
                return x.sum()
            elif x.sum() < mid:
                return fn(x, l, mid)
            else:
                return fn(x, mid + 1, r)

        def guard_filter_fn(guards):
            return [
                guard.guard_type not in ("CLOSURE_MATCH", "FUNCTION_MATCH")
                for guard in guards
            ]

        package = _CompilePackage(backend_type="eager")
        compiled_fn = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=guard_filter_fn
        )(fn)

        N = 10
        args_list = [(torch.tensor(x), 0, N - 1) for x in range(N)]
        for args in args_list:
            compiled_fn(*args)

        package.save(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            for args in args_list:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Detected recompile when torch.compile stance is 'fail_on_recompile'",
                ):
                    compiled_fn(*args)

        package.load(self.path())
        for args in args_list:
            self.assertEqual(compiled_fn(*args), args[0].sum())

        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                RuntimeError,
                "Detected recompile when torch.compile stance is 'fail_on_recompile'",
            ):
                compiled_fn(torch.tensor(N), 0, N - 1)

    def test_guard_invalidation(self):
        raise NotImplementedError("")

    def test_backward(self):
        raise NotImplementedError("")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
