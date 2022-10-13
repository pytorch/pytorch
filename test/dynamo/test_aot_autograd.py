# Owner(s): ["module: dynamo"]
import functools

import torch

import torch._dynamo
from torch._dynamo.optimizations.training import is_aot_autograd_safe_to_run
from torch._dynamo.testing import rand_strided


def compiler_safe_fn(gm, example_inputs, is_safe):
    is_safe[0] = is_aot_autograd_safe_to_run(gm, example_inputs)
    return gm.forward


class AotAutogradFallbackTests(torch._dynamo.testing.TestCase):
    def test_LSTM(self):
        # https://github.com/pytorch/torchdynamo/issues/1147
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_mod_model_lstm_lstm = torch.nn.LSTM(
                    64, 64, num_layers=2, bidirectional=True
                )

            def forward(self, permute: torch.Tensor):
                self_mod_model_lstm_lstm = self.self_mod_model_lstm_lstm(permute)
                return (self_mod_model_lstm_lstm,)

        is_safe = [True]
        mod = Repro()
        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_mod = torch._dynamo.optimize(compiler_fn)(mod)

        args = [((92, 4, 64), (1, 5888, 92), torch.float32, "cpu", False)]
        args = [
            rand_strided(sh, st, dt, dev).requires_grad_(rg)
            for (sh, st, dt, dev, rg) in args
        ]

        aot_mod(*args)
        self.assertTrue(not is_safe[0])

    def test_mutation(self):
        # https://github.com/pytorch/torchdynamo/issues/1301
        def fn(param, y):
            prev_grad = torch.is_grad_enabled()
            try:
                torch.set_grad_enabled(False)
                param.add_(y)
            finally:
                torch.set_grad_enabled(prev_grad)
            return y

        y = torch.randn(4)
        x = torch.nn.Parameter(torch.randn(4))
        is_safe = [True]
        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_fn = torch._dynamo.optimize(compiler_fn)(fn)
        aot_fn(x, y)
        self.assertTrue(not is_safe[0])

    def test_negative_testing(self):
        def fn(x, y):
            return torch.sin(x).add_(y)

        y = torch.randn(4)
        x = torch.randn(4)
        is_safe = [True]
        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_fn = torch._dynamo.optimize(compiler_fn)(fn)
        aot_fn(x, y)
        self.assertTrue(is_safe[0])


if __name__ == "__main__":
    from torch._dynamo.testing import run_tests

    run_tests()
