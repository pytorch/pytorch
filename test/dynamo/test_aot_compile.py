# Owner(s): ["module: dynamo"]

import functools
import inspect
import os
import pickle
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.aot_compile import ModelInput, SerializableCallable
from torch._dynamo.exc import PackageError, Unsupported
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.fx._graph_pickler import GraphPickler
from torch.testing._internal.common_utils import instantiate_parametrized_tests


MY_LAMBDA = lambda x: x + 1  # noqa: E731


class CustomCompiledFunction(torch._dynamo.aot_compile.SerializableCallable):
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        self.gm = gm
        self.example_inputs = example_inputs

    @classmethod
    def serialize_compile_artifacts(cls, fn) -> bytes:
        import sympy

        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import Options

        state = fn.__dict__.copy()
        graph_reducer_override = GraphPickler.reducer_override

        def _graph_reducer_override(self, obj):
            if (
                inspect.isclass(obj)
                and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")
            ):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return graph_reducer_override(self, obj)

        with patch.object(GraphPickler, "reducer_override", _graph_reducer_override):
            state["gm"] = GraphPickler.dumps(state["gm"], Options(ops_filter=None))
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


class RepeatInterleaveModule(torch.nn.Module):
    def forward(self, x):
        chunk = x.chunk(2, dim=-1)
        y = chunk[0]
        y_repeat = y.repeat_interleave(2, dim=-1)
        return y_repeat


class MultiModalMixin(torch.nn.Module):
    def forward(self, x):
        return super().forward(x)


class TextModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


class TestVLLMModel(MultiModalMixin, TextModel):
    def forward(self, x):
        return super().forward(x)


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

    def test_aot_compile_repeat_interleave(self):
        mod = RepeatInterleaveModule()

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        inputs = (torch.randn(2, 4),)

        # The first dim should be dynamic to repro the issue of repeat_interleave
        # torch._dynamo.mark_dynamic(inputs[0], [0])

        compiled_fn = torch.compile(
            mod,
            fullgraph=True,
            backend=backend,
        ).forward.aot_compile((inputs, {}))

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

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                foo,
                fullgraph=True,
                backend=backend,
            ).aot_compile((example_inputs, {}))
            actual = compiled_fn(*example_inputs)
            self.assertEqual(expected, actual)

    def test_decorated_function_with_functools_wrap_aot(self):
        def check_inputs(fn):
            @functools.wraps(fn)
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

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                foo,
                fullgraph=True,
                backend=backend,
            ).aot_compile((example_inputs, {}))
            actual = compiled_fn(*example_inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_disable_guard_check(self):
        def fn(x, y):
            return x + y

        with torch.no_grad():
            compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
                ((torch.randn(3, 4), torch.randn(3, 4)), {})
            )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        with self.assertRaisesRegex(RuntimeError, "GuardManager check failed"):
            compiled_fn(*inputs)
        compiled_fn.disable_guard_check()
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    def test_aot_compile_source_info(self):
        from torch._dynamo.package import SourceInfo

        def fn(x, y):
            return MY_LAMBDA(x) + y

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )

        source_info = compiled_fn.source_info()
        self.assertIsInstance(source_info, SourceInfo)
        self.assertEqual(len(source_info.inlined_sources), 2)
        self.assertEqual(next(iter(source_info.inlined_sources)).module, __name__)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        source_info = compiled_fn.source_info()
        self.assertIsInstance(source_info, SourceInfo)
        self.assertEqual(len(source_info.inlined_sources), 2)
        self.assertEqual(next(iter(source_info.inlined_sources)).module, __name__)

    def test_aot_compile_graph_break_error_fmt(self):
        def foo(x, y):
            a = x + x
            torch._dynamo.graph_break()
            b = y + y
            c = a + b
            return c

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(foo, fullgraph=True).aot_compile(
                ((torch.ones(3), torch.ones(3)), {})
            ),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_aot_compile.py", line N, in foo
    torch._dynamo.graph_break()""",
        )

    def test_guard_filter_override_aot(self):
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
        expected = foo(*example_inputs)  # noqa: F841

        def backend(gm, example_inputs):
            return CustomCompiledFunction(gm, example_inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                PackageError,
                "CLOSURE_MATCH guard cannot be serialized.",
            ):
                compiled_fn = torch.compile(  # noqa: F841
                    foo,
                    fullgraph=True,
                    backend=backend,
                    options={
                        "guard_filter_fn": lambda guard_entries: [
                            True for g in guard_entries
                        ]
                    },
                ).aot_compile((example_inputs, {}))

    def test_aot_compile_basic_fn_inductor(self):
        def fn(x, y):
            return x + y

        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor").aot_compile(
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

    def test_aot_compile_module(self):
        mod = SimpleLinearModule()

        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        )

        @contextmanager
        def train_mode(model):
            """
            Context manager that sets the model to training mode before entering the context.
            """
            model.train()
            yield

        @contextmanager
        def eval_mode(model):
            """
            Context manager that sets the model to evaluation mode before entering the context.
            """
            model.eval()
            yield

        inputs = [
            ModelInput(
                args=(torch.randn(3, 3),),
                kwargs={},
                contexts=[torch.no_grad(), eval_mode(model)],
            ),
            ModelInput(
                args=(torch.randn(3, 3),), kwargs={}, contexts=[train_mode(model)]
            ),
        ]
        assert isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
        model._aot_compile(
            inputs,
        )
        with torch.compiler.set_stance("fail_on_recompile"):
            model.eval()
            inputs = (torch.randn(3, 3),)
            expected = mod(*inputs)
            actual = model(*inputs)
            self.assertEqual(expected, actual)

            # Shouldn't recompile
            model.train()
            expected.sum().backward()

        model._save_aot_compiled_module(self.path())
        torch._dynamo.reset()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        )
        assert isinstance(model, torch._dynamo.eval_frame.OptimizedModule)
        with open(self.path(), "rb") as f:
            data = f.read()
            model._load_aot_compiled_module(data)

        with torch.compiler.set_stance("fail_on_recompile"):
            model.eval()
            inputs = (torch.randn(3, 3),)
            expected = mod(*inputs)
            actual = model(*inputs)
            self.assertEqual(expected, actual)

            # Shouldn't recompile
            model.train()
            expected.sum().backward()

    def test_aot_module_simplified_serializable_autograd(self):
        mod = SimpleLinearModule()
        compiled_fn: SerializableCallable = torch.compile(
            mod, fullgraph=True, backend="inductor"
        ).forward.aot_compile(((torch.randn(3, 3),), {}))
        backend_result = compiled_fn._artifacts.compiled_fn
        self.assertTrue(
            isinstance(
                backend_result,
                torch._dynamo.aot_compile.BundledAOTAutogradSerializableCallable,
            )
        )
        assert hasattr(backend_result.compiled_fn, "serialize")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

    def test_aot_module_simplified_serializable_inference(self):
        def fn(x):
            return x.sin()

        compiled_fn: SerializableCallable = torch.compile(
            fn, fullgraph=True, backend="inductor"
        ).aot_compile(((torch.randn(3, 3),), {}))
        backend_result = compiled_fn._artifacts.compiled_fn
        self.assertTrue(
            isinstance(
                backend_result,
                torch._dynamo.aot_compile.BundledAOTAutogradSerializableCallable,
            )
        )
        assert hasattr(backend_result.compiled_fn, "serialize")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

    def test_fullgraph_capture_with_pytree_module(self):
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            def forward(self, x):
                return {
                    "y": self.linear2(x[2] + 1),
                    "z": self.linear3(x[1] - 1),
                    "w": self.linear(x[0]["b"] + 2),
                    "v": self.linear1(x[0]["a"] - 2),
                }

        mod = Module()
        compiled_mod = dynamo_graph_capture_for_export(mod)(
            (
                {"a": torch.randn(3, 3), "b": torch.randn(3, 3)},
                torch.randn(3, 3),
                torch.randn(3, 3),
            )
        )

        inputs = (
            {"a": torch.randn(3, 3), "b": torch.randn(3, 3)},
            torch.randn(3, 3),
            torch.randn(3, 3),
        )
        self.assertEqual(compiled_mod(inputs), mod(inputs))

    def test_fullgraph_capture_with_pytree_func(self):
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        def foo(x):
            return {
                "y": x[2] + 1,
                "z": x[1] - 1,
                "w": x[0]["b"] + 2,
                "v": x[0]["a"] - 2,
            }

        compiled_foo = dynamo_graph_capture_for_export(foo)(
            (
                {"a": torch.randn(4, 3), "b": torch.randn(3, 2)},
                torch.randn(2, 3),
                torch.randn(3, 4),
            )
        )

        inputs = (
            {"a": torch.randn(4, 3), "b": torch.randn(3, 2)},
            torch.randn(2, 3),
            torch.randn(3, 4),
        )
        self.assertEqual(compiled_foo(inputs), foo(inputs))

    def test_aot_compile_with_closure_save_and_load(self):
        tmp = 2

        def fn(x, y):
            return x + y + tmp

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    def test_aot_compile_with_super_call(self):
        fn = TestVLLMModel()
        compiled_fn = torch.compile(fn.forward, fullgraph=True).aot_compile(
            ((torch.randn(3, 4),), {})
        )
        self.assertEqual(fn.forward.__code__.co_freevars, ("__class__",))
        inputs = (torch.randn(3, 4),)
        expected = fn(*inputs)
        actual = compiled_fn(fn, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(fn, *inputs)
        self.assertEqual(expected, actual)

    def test_aot_compile_with_default_args(self):
        def fn(x, y=1):
            return x + x

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(
            ((torch.randn(3, 4),), {})
        )
        inputs = (torch.randn(3, 4),)
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
