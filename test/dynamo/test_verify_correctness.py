# Owner(s): ["module: dynamo"]
import operator

import torch
import torch._dynamo
import torch._dynamo.config as config
import torch._dynamo.test_case
from torch._dynamo.aot_compile_types import GraphModuleSerializableCallable
from torch._dynamo.backends.debugging import eager
from torch._dynamo.output_graph import WrapperBackend
from torch._dynamo.testing import same
from torch.fx._lazy_graph_module import _force_skip_lazy_graph_module


class Seq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def transform(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # operator.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if node.target == operator.mul:
                node.target = operator.add

    gm.graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.

    gm.recompile()
    return gm


@config.patch("verify_correctness", True)
class TestVerifyCorrectness(torch._dynamo.test_case.TestCase):
    def test_preserves_user_modules_during_verification(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = torch.nn.Identity()

            def forward(self, x):
                return self.inner(x)

        gm = torch.fx.symbolic_trace(Mod())
        seen = []
        handle = torch.nn.modules.module.register_module_forward_pre_hook(
            lambda module, args: seen.append(
                torch._dynamo.utils.is_dynamo_runtime_module(module)
            )
            if isinstance(module, torch.nn.Identity)
            else None
        )
        try:
            WrapperBackend(eager)(gm, [torch.randn(2, 2)])
        finally:
            handle.remove()

        self.assertEqual(seen, [False, False])

    def test_preserves_serializable_backend_callable(self):
        gm = torch.fx.symbolic_trace(lambda x: x.sin())
        x = torch.randn(2, 2)
        backend = WrapperBackend(eager)

        with torch._functorch.config.patch(force_autograd_cache=True):
            compiled_fn = backend(gm, [x])

        self.assertIsInstance(compiled_fn, GraphModuleSerializableCallable)
        self.assertTrue(
            any(
                ref() is compiled_fn.graph_module for ref in backend.runtime_module_refs
            )
        )

    def test_example_inputs(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)

        self.assertEqual(r1.shape, r2.shape)
        self.assertEqual(r1.shape, r3.shape)
        self.assertEqual(r1.device, r2.device)
        self.assertEqual(r1.device, r3.device)

    @_force_skip_lazy_graph_module()
    def test_torchscript(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        opt_s = torch.compile(s, backend="ts")
        r2 = opt_s(i)
        self.assertTrue(same(r1, r2))

    def test_incorrect_verify_true(self):
        """
        If a bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=True, it will
        check the correctness of outputs and raise an error
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        toy_example(i1, i2)
        try:
            opt_toy_example = torch.compile(toy_example, backend=incorrect_compile_fn)
            opt_toy_example(i1, i2)
        except RuntimeError:
            pass
        else:
            self.fail("expected failure")

    @config.patch("verify_correctness", False)
    def test_incorrect_verify_false(self):
        """
        The bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=False, wrong outputs
        will return
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        r1 = toy_example(i1, i2)
        opt_toy_example = torch.compile(toy_example, backend=incorrect_compile_fn)
        r2 = opt_toy_example(i1, i2)
        self.assertTrue(not same(r1, r2))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
