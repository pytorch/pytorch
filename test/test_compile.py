# Owner(s): ["module: dynamo"]

import torch
from torch._dynamo.testing import CompileCounter
import tempfile
import os
import unittest

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

class TorchCompileTests(unittest.TestCase):
    def test_compilation(self):
        model = ToyModel()
        cnt = CompileCounter()
        # opt_mod = torch._dynamo.optimize(cnt)(model)
        model.compile(backend=cnt)
        x = torch.randn(10, 10)
        model(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_overwrite_call_impl(self):
        model = ToyModel()
        model(torch.randn(1, 10))
        self.assertTrue(model._compiled_call_impl is None)
        model.compile()
        model(torch.rand(1, 10))
        self.assertTrue(model._compiled_call_impl is not None)

    def test_compiled_model_can_be_saved_and_loaded(self):
        model = ToyModel()
        model.compile()
        model(torch.randn(1, 10))

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model, os.path.join(tmpdirname, "model.pt"))
            torch.load(os.path.join(tmpdirname, "model.pt"))

        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.save(model.state_dict(), os.path.join(tmpdirname, "model.pt"))
            loaded_model = ToyModel()
            loaded_model.load_state_dict(torch.load(os.path.join(tmpdirname, "model.pt")))
