import os
import sys
import torch
from torch.testing._internal.jit_utils import JitTestCase, make_global
from torch.jit._monkeytype_config import _IS_MONKEYTYPE_INSTALLED
from typing import List, Dict, Tuple, Any  # noqa: F401

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if not _IS_MONKEYTYPE_INSTALLED:
    print("monkeytype is not installed. Skipping tests for Profile-Directed Typing", file=sys.stderr)
    JitTestCase = object  # type: ignore[misc, assignment] # noqa: F811

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestPDT(JitTestCase):
    """
    A suite of tests for profile directed typing in TorchScript.
    """
    def test_nn_module(self):
        class TestPDTModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x) -> Any:
                if isinstance(x, int):
                    return x + 1
                elif isinstance(x, float):
                    return x - 1
                else:
                    return x

        make_global(TestPDTModel)
        pdt_model = TestPDTModel()
        scripted_pdt_model = torch.jit._script_pdt(pdt_model, example_inputs=[(10, ), (10.80, ), (False, )])
        self.assertEqual(scripted_pdt_model(50), pdt_model(50))
        self.assertEqual(scripted_pdt_model(1.8), pdt_model(1.8))
        self.assertTrue(scripted_pdt_model(True), pdt_model(True))

    def test_nested_nn_module_class(self):
        class NestedPDTInner(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                if isinstance(x, int):
                    return x * 10
                return x

        class NestedModulePDTWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x)

        make_global(NestedPDTInner, NestedModulePDTWrapper)
        inner_pdt_model = NestedPDTInner()
        wrapped_pdt_model = NestedModulePDTWrapper(inner_pdt_model)
        scripted_pdt_model = torch.jit._script_pdt(wrapped_pdt_model, example_inputs=[(20, ), (2.7, ), (False, )])
        self.assertEqual(scripted_pdt_model(30), wrapped_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), wrapped_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), wrapped_pdt_model(True))

    def test_nested_function_in_forward(self):
        class NestedFunctionInForward(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return self.fun(x) + 10

            def fun(self, x):
                if isinstance(x, bool):
                    return 0
                elif isinstance(x, int):
                    return x + 1
                return 0

        make_global(NestedFunctionInForward)
        pdt_model = NestedFunctionInForward()
        scripted_pdt_model = torch.jit._script_pdt(pdt_model, example_inputs=[(20, ), (False, )])
        self.assertEqual(scripted_pdt_model(30), pdt_model(30))
        self.assertEqual(scripted_pdt_model(True), pdt_model(True))
