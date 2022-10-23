# Owner(s): ["module: dynamo"]
import functools

import torch

import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.optimizations.training import is_aot_autograd_safe_to_run
from torch._dynamo.testing import CompileCounter, rand_strided


def compiler_safe_fn(gm, example_inputs, is_safe):
    is_safe[0] = is_aot_autograd_safe_to_run(gm, example_inputs)
    return gm.forward


class AotAutogradFallbackTests(torch._dynamo.test_case.TestCase):
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

    def test_mutation1(self):
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, 256, None),
                    slice(None, 257, None),
                )
            ]
            _stack0[
                (
                    slice(None, None, None),
                    slice(None, -1, None),
                    slice(None, None, None),
                    slice(256, None, None),
                )
            ] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)

        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        is_safe = [True]
        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_fn = torch._dynamo.optimize(compiler_fn)(fn)
        aot_fn(x, y)
        self.assertTrue(not is_safe[0])

    def test_negative_testing_mutation(self):
        def fn(_stack0: torch.Tensor, diagonal_chunked_attention_scores: torch.Tensor):
            getitem = diagonal_chunked_attention_scores[
                (
                    slice(None, None, None),
                    slice(None, None, None),
                    slice(None, 256, None),
                    slice(None, 257, None),
                )
            ]
            _stack0 = torch.sin(_stack0)
            _stack0[
                (
                    slice(None, None, None),
                    slice(None, -1, None),
                    slice(None, None, None),
                    slice(256, None, None),
                )
            ] = getitem
            view = _stack0.view(1, 12, 1024, 513)
            return (view,)

        x = torch.randn(torch.Size([12, 4, 256, 513]))
        y = torch.randn(torch.Size([12, 3, 512, 513]))
        is_safe = [True]
        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_fn = torch._dynamo.optimize(compiler_fn)(fn)
        aot_fn(x, y)
        self.assertTrue(is_safe[0])

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

    def test_call_fn_with_non_const_inputs_aot_safe(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self):
                super(ModuleSpecialFwd, self).__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=20, kernel_size=(5, 5)
                )

            def _conv_forward(self, x):
                return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)

            def forward(self, x):
                return self._conv_forward(x)

        # Init mod
        mod = ModuleSpecialFwd()
        rx = torch.randn([3, 10, 10])

        # Run it for real
        real = mod(rx)

        # Run it in export
        graph, _ = torch._dynamo.export(mod, rx)

        # Run exported graph with AOT
        is_safe = [True]
        self.assertTrue(torch._dynamo.testing.same(real, graph(rx)))

        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_fn = torch._dynamo.optimize(compiler_fn)(graph)
        aot_fn(rx)
        self.assertTrue(is_safe[0])

    def test_call_fn_with_non_const_inputs_aot_unsafe(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self):
                super(ModuleSpecialFwd, self).__init__()

            def _some_bad_fwd(self, param, y):
                prev_grad = torch.is_grad_enabled()
                try:
                    torch.set_grad_enabled(False)
                    param.add_(y)
                finally:
                    torch.set_grad_enabled(prev_grad)
                return y

            def forward(self, x, y):
                return self._some_bad_fwd(x, y)

        # Init mod
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn(4))
        y = torch.randn([4])

        # Run it for real
        real = mod(x, y)

        # Run it in export
        graph, _ = torch._dynamo.export(mod, x, y)

        # Assert equal
        self.assertTrue(torch._dynamo.testing.same(real, graph(x, y)))

        # Run exported graph with AOT
        is_safe = [True]

        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_fn = torch._dynamo.optimize(compiler_fn)(graph)
        aot_fn(x, y)
        self.assertTrue(not is_safe[0])

    def test_call_fn_with_non_const_inputs_aot_unsafe_control_flow(self):
        class ModuleSpecialFwd(torch.nn.Module):
            def __init__(self):
                super(ModuleSpecialFwd, self).__init__()

            def _some_bad_fwd(self, param, y):
                if y[0][0] < 3:
                    return y + param
                return param * y

            def forward(self, x, y):
                a = x * y
                a = self._some_bad_fwd(a, a)
                b = x + y
                return a * b

        # Init mod
        mod = ModuleSpecialFwd()
        x = torch.nn.Parameter(torch.randn([2, 2]))
        y = torch.randn([2, 2])

        # Run it for real
        real = mod(x, y)

        # Run it through optimize, with our capturing fn

        gms = []
        counter = CompileCounter()

        def capturing_fn(gm, inputs):
            nonlocal gms
            gms.append(gm)
            return counter(gm, inputs)

        optimized_mod = torch._dynamo.optimize(capturing_fn)(mod)

        # Assert equal
        self.assertTrue(torch._dynamo.testing.same(real, optimized_mod(x, y)))

        # Uncomment to reproduce commented out graphs below.
        # for gm in gms:
        #     print("GM CODE", gm.code)

        self.assertEqual(counter.frame_count, 4)
        self.assertEqual(counter.op_count, 7)
        # Graph 1
        # def forward(self, x : torch.nn.parameter.Parameter, y : torch.Tensor):
        #     mul = x * y;  x = y = None
        #     return (mul,)
        # BREAK
        # Graph 2
        # def forward(self, y : torch.Tensor):
        #     getitem = y[0];  y = None
        #     getitem_1 = getitem[0];  getitem = None
        #     lt = getitem_1 < 3;  getitem_1 = None
        #     return (lt,)
        # BREAK
        # Graph 3
        # def forward(self, param : torch.Tensor, y : torch.Tensor):
        #     add = y + param;  y = param = None
        #     return (add,)
        # BREAK
        # Graph 4
        # def forward(self, _stack0 : torch.Tensor, x : torch.nn.parameter.Parameter, y : torch.Tensor):
        #     add = x + y;  x = y = None
        #     mul = _stack0 * add;  _stack0 = add = None
        #     return (mul,)

        # Run fn with AOT
        torch._dynamo.reset()
        is_safe = [True]

        compiler_fn = functools.partial(compiler_safe_fn, is_safe=is_safe)
        aot_fn = torch._dynamo.optimize(compiler_fn)(optimized_mod)
        aot_fn(x, y)
        self.assertTrue(is_safe[0])


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
