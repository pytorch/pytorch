# Owner(s): ["module: dynamo"]

import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.compiler
from torch._dynamo import reset
from torch._dynamo.backends.debugging import explain
from torch._dynamo.testing import CompileCounter


class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def fn3(x):
    x = torch.sin(x)
    torch._dynamo.graph_break()
    x = torch.sin(x)
    return x


def fn2(x):
    x = torch.cos(x)
    x = fn3(x)
    x = torch.cos(x)
    return x


def fn1(x):
    x = torch.tan(x)
    torch._dynamo.graph_break()
    x = fn2(x)
    x = torch.tan(x)
    return x


def fn(x):
    x = torch.sigmoid(x)
    x = fn1(x)
    x = torch.sigmoid(x)
    return x


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


class ExplainCompilerTests(unittest.TestCase):
    @mock.patch("builtins.print")
    def test_explain_print_call_graph_nodes(self, mock_print):
        reset()
        torch._dynamo.config.explain_print_graphs = True
        opt_fn = torch._dynamo.optimize(explain)(fn)
        self.assertAlmostEqual(0.7581, opt_fn(torch.ones([1])).item(), places=4)
        self.assertEqual(12, mock_print.call_count)
        torch._dynamo.config.explain_print_graphs = False

    @mock.patch("builtins.print")
    def test_explain_print_call_break_reasons(self, mock_print):
        reset()
        opt_fn = torch._dynamo.optimize(explain)(fn)
        self.assertAlmostEqual(0.7581, opt_fn(torch.ones([1])).item(), places=4)
        self.assertEqual(4, mock_print.call_count)

    @mock.patch("builtins.print")
    def test_explain_print_call_registration(self, mock_print):
        reset()
        opt_fn = torch._dynamo.optimize("explain")(fn)
        self.assertAlmostEqual(0.7581, opt_fn(torch.ones([1])).item(), places=4)
        self.assertEqual(4, mock_print.call_count)
