# Owner(s): ["module: dynamo"]
import torch
from torch.testing._internal.jit_utils import JitTestCase


class TestTracer(JitTestCase):
    def test_jit_save(self):
        def fn():
            class Foo(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.a = 3

                @torch.jit.export
                def __getstate__(self):
                    return (3, self.training)

                @torch.jit.export
                def __setstate__(self, state):
                    self.a = state[0]
                    self.training = state[1]

                def forward(self, x):
                    return x + self.a

            f = Foo()

            return torch.jit.trace(f, (torch.rand(3, 4),))

        fn()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn()

    def test_jit_trace_errors(self):
        @torch.compile(backend="eager", dynamic=True)
        def f(x):
            return x + 1

        with self.assertRaises(RuntimeError):
            torch.jit.trace(f, torch.randn(3))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
