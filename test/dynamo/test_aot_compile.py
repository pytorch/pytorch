# Owner(s): ["module: dynamo"]

import os
import pickle

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.fx._graph_pickler import GraphPickler
from torch.testing._internal.common_utils import instantiate_parametrized_tests


class CustomCompiledFunction(torch._dynamo.aot_compile.SerializableCallable):
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs

    @classmethod
    def serialize_compile_artifacts(cls, fn) -> bytes:
        state = fn.__dict__.copy()
        state["gm"] = GraphPickler.dumps(state["gm"])
        return pickle.dumps(state)

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes):
        state = pickle.loads(data)
        fake_mode = torch._subclasses.FakeTensorMode()
        state["gm"] = GraphPickler.loads(state["gm"], fake_mode)
        state["gm"].recompile()
        return cls(**state)

    def __call__(self, *args, **kwargs):
        return self.gm(*args, **kwargs)


class SimpleLinearModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


@torch._dynamo.config.patch("enable_aot_compile", True)
@instantiate_parametrized_tests
class TestAOTCompile(torch._inductor.test_case.TestCase):
    def path(self):
        path = os.path.join(cache_dir(), f"package_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "model.pt")

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()

    def test_aot_compile_basic_fn(self):
        def fn(x, y):
            return x + y

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        compiled_fn = torch.compile(fn, fullgraph=True, backend=backend).aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_basic_forward(self):
        mod = SimpleLinearModule()

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        compiled_fn = torch.compile(
            mod,
            fullgraph=True,
            backend=backend,
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        ).forward.aot_compile(((torch.randn(3, 3),), {}))
        inputs = (torch.randn(3, 3),)
        expected = mod(*inputs)
        actual = compiled_fn(mod, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(mod, *inputs)
            self.assertEqual(expected, actual)

    def test_decorated_function_aot(self):
        def check_inputs(fn):
            def _fn(*args, **kwargs):
                for arg in args:
                    assert arg.shape[0] > 1

                return fn(*args, **kwargs)

            return _fn

        @check_inputs
        def foo(x, y):
            a = x + x
            b = y + y
            c = a + b
            return c

        example_inputs = (torch.ones(3), torch.ones(3))
        expected = foo(*example_inputs)

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        def skip_closure_match_guards(guard_entries):
            return [g.guard_type != "CLOSURE_MATCH" for g in guard_entries]

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                foo,
                fullgraph=True,
                backend=backend,
                options={
                    "guard_filter_fn": skip_closure_match_guards,
                },
            ).aot_compile((example_inputs, {}))
            actual = compiled_fn(*example_inputs)
            self.assertEqual(expected, actual)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
