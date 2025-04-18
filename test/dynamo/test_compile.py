# Owner(s): ["module: dynamo"]

import inspect
import io
import os
import tempfile
from unittest.mock import patch

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class InPlaceCompilationTests(TestCase):
    def test_compilation(self):
        torch._dynamo.reset()
        model = ToyModel()
        cnt = CompileCounter()
        model.compile(backend=cnt)
        x = torch.randn(10, 10)
        model(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_overwrite_call_impl(self):
        torch._dynamo.reset()
        model = ToyModel()
        self.assertTrue(model._compiled_call_impl is None)
        model.compile()
        self.assertTrue(model._compiled_call_impl is not None)

    def test_save(self):
        torch._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(torch.randn(1, 10))

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model, os.path.join(tmpdirname, "model.pt"))
            # weights_only=False as this is a legacy use case that loads a module
            loaded_model = torch.load(
                os.path.join(tmpdirname, "model.pt"), weights_only=False
            )
            loaded_model(torch.randn(1, 10))

    def test_state_dict_save(self):
        torch._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(torch.randn(1, 10))
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model.state_dict(), os.path.join(tmpdirname, "model.pt"))
            loaded_model = ToyModel()
            loaded_model.load_state_dict(
                # weights_only=False as this is a legacy use case that loads a module
                torch.load(os.path.join(tmpdirname, "model.pt"), weights_only=False)
            )
            loaded_model(torch.randn(1, 10))

    def test_jit_save(self):
        torch._dynamo.reset()
        model = ToyModel()
        model.compile()
        model(torch.randn(1, 10))
        scripted_model = torch.jit.script(model)
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.jit.save(scripted_model, os.path.join(tmpdirname, "model.pt"))
            loaded_model = torch.jit.load(os.path.join(tmpdirname, "model.pt"))
            loaded_model(torch.randn(1, 10))

    def test_compilation_callback(self):
        torch._dynamo.reset()

        @torch._dynamo.on_compile_start
        def start_callback():
            print("Compilation started.")

        @torch._dynamo.on_compile_end
        def end_callback():
            print("Compilation ended.")

        mod = ToyModel()
        x = torch.randn(10, 10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_mod = torch.compile(backend="eager", fullgraph=True)(mod)
            opt_mod(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(printed_output, "Compilation started.\nCompilation ended.")

    def test_compile_eager_options(self):
        @torch.compile(backend="eager", options={"foo": 2})
        def f(x):
            return x + x

        f(torch.randn(3))

        @torch.compile(backend="aot_eager", options={"foo": 2})
        def g(x):
            return x + x

        g(torch.randn(3))

    def test_compilation_callback_with_graph_break(self):
        torch._dynamo.reset()
        counter = 0

        @torch._dynamo.on_compile_start
        def start_callback():
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        @torch._dynamo.on_compile_end
        def end_callback():
            nonlocal counter
            counter += 1
            print(f"Counter = {counter}")

        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            torch._dynamo.graph_break()
            return torch.sin(x)

        x = torch.randn(10, 10)

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            fn(x)
            printed_output = mock_stdout.getvalue().strip()

        self.assertEqual(
            printed_output, "Counter = 1\nCounter = 2\nCounter = 3\nCounter = 4"
        )

    def test_compilation_constant_hasattr_fail(self):
        @torch.compile(backend="eager")
        def fn(x):
            return x.max()

        # We should fallback to normal mode, and throw a AttributeError, not a internal dynamo exception
        with self.assertRaises(AttributeError):
            fn(None)

    def test_compilation_evnum_hasattr_fail(self):
        from enum import Enum

        class TestEnum(Enum):
            VALID = 1

        @torch.compile(backend="eager")
        def fn(x):
            return x.max()

        # We should fallback to normal mode, and throw a AttributeError, not a internal dynamo exception
        with self.assertRaises(AttributeError):
            fn(TestEnum.VALID)

    def test_compilation_name_error(self):
        @torch.compile(backend="eager")
        def fn(x):
            x = x + 1
            does_not_exist()  # noqa: F821
            return x

        x = torch.randn(10, 10)
        with self.assertRaises(NameError):
            fn(x)

    def test_compilation_tensor_invalid_method(self):
        @torch.compile(backend="eager")
        def fn(x):
            y = torch.tensor(x)
            return y.doesnotexist()

        x = torch.randn(10, 10)

        with self.assertRaises(AttributeError):
            fn(x)

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_compilation_nn_module_invalid_method(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + self.doesnotexist

        mod = Mod()
        opt_mod = torch.compile(mod, backend="eager")
        x = torch.randn(1, 1)
        with self.assertRaises(AttributeError):
            opt_mod(x)

    def test_torch_script_compilation(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x

        a = torch.randn(1, 1)
        out = torch.compile(fn)(a)
        self.assertEqual(out, a)


# The private variants of the below functions are extensively tested
# So as long as the signatures match we're good
class PublicTorchCompilerTests(TestCase):
    def check_signature(self, public_fn_name, private_fn_name, private_namespace):
        public_fn = getattr(torch.compiler, public_fn_name)
        private_fn = getattr(private_namespace, private_fn_name)

        public_sig = inspect.signature(public_fn)
        private_sig = inspect.signature(private_fn)

        self.assertEqual(
            public_sig,
            private_sig,
            f"Signatures do not match for function {public_fn_name}() \n Public: {public_sig} \n Private: {private_sig}",
        )

    def test_dynamo_signatures(self):
        function_names = [
            "reset",
            "allow_in_graph",
            "list_backends",
            "assume_constant_result",
            "disable",
        ]

        for fn_name in function_names:
            self.check_signature(fn_name, fn_name, torch._dynamo)


if __name__ == "__main__":
    run_tests()
