# Owner(s): ["module: dynamo"]

import inspect
import os
import tempfile
import unittest

import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounter


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class InPlaceCompilationTests(unittest.TestCase):
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
            loaded_model = torch.load(os.path.join(tmpdirname, "model.pt"))
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
                torch.load(os.path.join(tmpdirname, "model.pt"))
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


# The private variants of the below functions are extensively tested
# So as long as the signatures match we're good
class PublicTorchCompilerTests(unittest.TestCase):
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
