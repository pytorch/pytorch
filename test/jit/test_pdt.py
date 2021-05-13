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
        inp: List[Tuple[Any, ...]] = [(20, ), (2.7, ), (False, ), ]
        scripted_pdt_model = torch.jit._script_pdt(pdt_model, example_inputs={pdt_model: inp})
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
        inp: List[Tuple[Any, ...]] = [(20, ), (False, )]
        scripted_pdt_model = torch.jit._script_pdt(wrapped_pdt_model, example_inputs={wrapped_pdt_model: inp})
        self.assertEqual(scripted_pdt_model(30), wrapped_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), wrapped_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), wrapped_pdt_model(True))

    def test_nested_nn_module_class_with_args(self):
        class NestedModulePDTInner(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                if isinstance(x, int):
                    return x * 10 + y
                return x

        class NestedModulePDTOuter(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x, 20)

        make_global(NestedModulePDTInner, NestedModulePDTOuter)
        inner_pdt_model = NestedModulePDTInner()
        outer_pdt_model = NestedModulePDTOuter(inner_pdt_model)
        inner_input: List[Tuple[Any, ...]] = [(10, 10), (1.9, 20), ]
        outer_input: List[Tuple[Any, ...]] = [(20, ), (False, )]
        scripted_pdt_model = torch.jit._script_pdt(outer_pdt_model, example_inputs={inner_pdt_model: inner_input,
                                                   outer_pdt_model: outer_input, })
        self.assertEqual(scripted_pdt_model(30), outer_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), outer_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), outer_pdt_model(True))

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
        inp: List[Tuple[Any, ...]] = [(-1, ), (False, )]
        scripted_pdt_model = torch.jit._script_pdt(pdt_model, example_inputs={pdt_model: inp})
        self.assertEqual(scripted_pdt_model(30), pdt_model(30))
        self.assertEqual(scripted_pdt_model(True), pdt_model(True))

    def test_nn_module_with_export_function(self):
        class TestModelWithExport(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @torch.jit.export
            def fn(self, x, y) -> Any:
                assert not (isinstance(x, bool) and isinstance(y, bool))
                if isinstance(x, int) and isinstance(y, int):
                    return x + y
                elif isinstance(x, float) and isinstance(y, float):
                    return x - y
                else:
                    return -1


        make_global(TestModelWithExport)
        pdt_model = TestModelWithExport()
        inp: List[Tuple[Any, ...]] = [(20, 10, ), (2.7, 8.9, ), ]
        scripted_pdt_model = torch.jit._script_pdt(pdt_model, example_inputs={pdt_model.fn: inp})
        self.assertEqual(scripted_pdt_model.fn(10, 90), pdt_model.fn(10, 90))
        self.assertEqual(scripted_pdt_model.fn(1.8, 2.2), pdt_model.fn(1.8, 2.2))
        self.assertTrue(scripted_pdt_model.fn(torch.ones(1), 2), pdt_model.fn(torch.ones(1), 2))

    def test_class_methods(self):
        class PDTModel:
            def test_sum(self, a):
                return sum(a)

        make_global(PDTModel)
        pdt_model = PDTModel()
        inp: List[Tuple[Any, ...]] = [([10, 20, ], ), ]
        scripted_pdt_model = torch.jit._script_pdt(PDTModel, example_inputs={pdt_model.test_sum: inp})
        script_model = scripted_pdt_model()
        self.assertEqual(script_model.test_sum([10, 20, 30, ], ), pdt_model.test_sum([10, 20, 30, ], ))

    def test_class_with_multiple_methods(self):
        class PDTModelWithManyMethods:
            def test_list_to_dict(self, a):
                new_dictionary: Dict[float, bool] = {}
                for element in a:
                    new_dictionary[element] = True
                return new_dictionary

            def test_substring(self, a, b):
                return b in a

        make_global(PDTModelWithManyMethods)
        pdt_model = PDTModelWithManyMethods()
        list_inp: List[Tuple[Any, ...]] = [([1.2, 2.3, ], ), ]
        str_inp: List[Tuple[Any, ...]] = [("abc", "b", ), ]
        scripted_pdt_model = torch.jit._script_pdt(PDTModelWithManyMethods, example_inputs={pdt_model.test_list_to_dict: list_inp,
                                                   pdt_model.test_substring: str_inp})
        script_model = scripted_pdt_model()
        self.assertEqual(script_model.test_list_to_dict([1.1, 2.2, 3.3, ], ), pdt_model.test_list_to_dict([1.1, 2.2, 3.3, ], ))
        self.assertEqual(script_model.test_substring("helloworld", "world", ), pdt_model.test_substring("helloworld", "world", ))
        self.assertEqual(script_model.test_substring("helloworld", "def", ), pdt_model.test_substring("helloworld", "def", ))

    def test_multiple_class_with_same_method(self):
        class PDTModelOne:
            def test_find(self, a, b):
                return b in a.keys()

        class PDTModelTwo:
            def test_find(self, a, b):
                return b in a

        make_global(PDTModelOne, PDTModelTwo)
        pdt_model_one = PDTModelOne()
        pdt_model_two = PDTModelTwo()
        dict_inp: List[Tuple[Any, ...]] = [({1.2: True, 2.3: False, }, 1.2), ]
        list_inp: List[Tuple[Any, ...]] = [(["abc", "b", ], "c"), ]
        scripted_pdt_model_one = torch.jit._script_pdt(PDTModelOne, example_inputs={pdt_model_one.test_find: dict_inp})
        scripted_pdt_model_two = torch.jit._script_pdt(PDTModelTwo, example_inputs={pdt_model_two.test_find: list_inp})

        script_model_one, script_model_two = scripted_pdt_model_one(), scripted_pdt_model_two()
        self.assertEqual(script_model_one.test_find({1.1: True, 2.2: True, 3.3: False, }, 4.4),
                         pdt_model_one.test_find({1.1: True, 2.2: True, 3.3: False, }, 4.4))
        self.assertEqual(script_model_two.test_find(["hello", "world", ], "world"),
                         pdt_model_two.test_find(["hello", "world", ], "world"))
