# Owner(s): ["oncall: jit"]

import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple  # noqa: F401

import torch
from torch.jit._monkeytype_config import _IS_MONKEYTYPE_INSTALLED
from torch.testing._internal.common_utils import NoTest
from torch.testing._internal.jit_utils import JitTestCase, make_global


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if not _IS_MONKEYTYPE_INSTALLED:
    print(
        "monkeytype is not installed. Skipping tests for Profile-Directed Typing",
        file=sys.stderr,
    )
    JitTestCase = NoTest  # type: ignore[misc, assignment] # noqa: F811

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
            def forward(self, x) -> Any:
                if isinstance(x, int):
                    return x + 1
                elif isinstance(x, float):
                    return x - 1
                else:
                    return x

        make_global(TestPDTModel)
        pdt_model = TestPDTModel()
        inp: List[Tuple[Any, ...]] = [
            (20,),
            (2.7,),
            (False,),
        ]
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model: inp}
        )
        self.assertEqual(scripted_pdt_model(50), pdt_model(50))
        self.assertEqual(scripted_pdt_model(1.8), pdt_model(1.8))
        self.assertTrue(scripted_pdt_model(True), pdt_model(True))

    def test_nested_nn_module_class(self):
        class NestedPDTInner(torch.nn.Module):
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
        inp: List[Tuple[Any, ...]] = [(20,), (False,)]
        scripted_pdt_model = torch.jit.script(
            wrapped_pdt_model, example_inputs={wrapped_pdt_model: inp}
        )
        self.assertEqual(scripted_pdt_model(30), wrapped_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), wrapped_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), wrapped_pdt_model(True))

    def test_nested_nn_module_class_with_args(self):
        class NestedModulePDTInner(torch.nn.Module):
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
        inner_input: List[Tuple[Any, ...]] = [
            (10, 10),
            (1.9, 20),
        ]
        outer_input: List[Tuple[Any, ...]] = [(20,), (False,)]
        scripted_pdt_model = torch.jit.script(
            outer_pdt_model,
            example_inputs={
                inner_pdt_model: inner_input,
                outer_pdt_model: outer_input,
            },
        )
        self.assertEqual(scripted_pdt_model(30), outer_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), outer_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), outer_pdt_model(True))

    def test_nested_function_in_forward(self):
        class NestedFunctionInForward(torch.nn.Module):
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
        inp: List[Tuple[Any, ...]] = [(-1,), (False,)]
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model: inp}
        )
        self.assertEqual(scripted_pdt_model(30), pdt_model(30))
        self.assertEqual(scripted_pdt_model(True), pdt_model(True))

    def test_nn_module_with_export_function(self):
        class TestModelWithExport(torch.nn.Module):
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
        inp: List[Tuple[Any, ...]] = [
            (
                20,
                10,
            ),
            (
                2.7,
                8.9,
            ),
        ]
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model.fn: inp}
        )
        self.assertEqual(scripted_pdt_model.fn(10, 90), pdt_model.fn(10, 90))
        self.assertEqual(scripted_pdt_model.fn(1.8, 2.2), pdt_model.fn(1.8, 2.2))
        self.assertTrue(
            scripted_pdt_model.fn(torch.ones(1), 2), pdt_model.fn(torch.ones(1), 2)
        )

    def test_class_methods(self):
        class PDTModel:
            def test_sum(self, a):
                return sum(a)

        make_global(PDTModel)
        pdt_model = PDTModel()
        inp: List[Tuple[Any, ...]] = [
            (
                [
                    10,
                    20,
                ],
            ),
        ]
        scripted_pdt_model = torch.jit.script(
            PDTModel, example_inputs={pdt_model.test_sum: inp}
        )
        script_model = scripted_pdt_model()
        self.assertEqual(
            script_model.test_sum(
                [
                    10,
                    20,
                    30,
                ],
            ),
            pdt_model.test_sum(
                [
                    10,
                    20,
                    30,
                ],
            ),
        )

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
        list_inp: List[Tuple[Any, ...]] = [
            (
                [
                    1.2,
                    2.3,
                ],
            ),
        ]
        str_inp: List[Tuple[Any, ...]] = [
            (
                "abc",
                "b",
            ),
        ]
        scripted_pdt_model = torch.jit.script(
            PDTModelWithManyMethods,
            example_inputs={
                pdt_model.test_list_to_dict: list_inp,
                pdt_model.test_substring: str_inp,
            },
        )
        script_model = scripted_pdt_model()
        self.assertEqual(
            script_model.test_list_to_dict(
                [
                    1.1,
                    2.2,
                    3.3,
                ],
            ),
            pdt_model.test_list_to_dict(
                [
                    1.1,
                    2.2,
                    3.3,
                ],
            ),
        )
        self.assertEqual(
            script_model.test_substring(
                "helloworld",
                "world",
            ),
            pdt_model.test_substring(
                "helloworld",
                "world",
            ),
        )
        self.assertEqual(
            script_model.test_substring(
                "helloworld",
                "def",
            ),
            pdt_model.test_substring(
                "helloworld",
                "def",
            ),
        )

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
        dict_inp: List[Tuple[Any, ...]] = [
            (
                {
                    1.2: True,
                    2.3: False,
                },
                1.2,
            ),
        ]
        list_inp: List[Tuple[Any, ...]] = [
            (
                [
                    "abc",
                    "b",
                ],
                "c",
            ),
        ]
        scripted_pdt_model_one = torch.jit.script(
            PDTModelOne, example_inputs={pdt_model_one.test_find: dict_inp}
        )
        scripted_pdt_model_two = torch.jit.script(
            PDTModelTwo, example_inputs={pdt_model_two.test_find: list_inp}
        )

        script_model_one, script_model_two = (
            scripted_pdt_model_one(),
            scripted_pdt_model_two(),
        )
        self.assertEqual(
            script_model_one.test_find(
                {
                    1.1: True,
                    2.2: True,
                    3.3: False,
                },
                4.4,
            ),
            pdt_model_one.test_find(
                {
                    1.1: True,
                    2.2: True,
                    3.3: False,
                },
                4.4,
            ),
        )
        self.assertEqual(
            script_model_two.test_find(
                [
                    "hello",
                    "world",
                ],
                "world",
            ),
            pdt_model_two.test_find(
                [
                    "hello",
                    "world",
                ],
                "world",
            ),
        )

    def test_pdt(self):
        def test_sum(a, b):
            return a + b

        make_global(test_sum)
        scripted_fn_add = torch.jit.script(test_sum, example_inputs=[(3, 4)])
        self.assertEqual(scripted_fn_add(10, 2), test_sum(10, 2))

        def test_sub(a, b):
            return a - b

        make_global(test_sub)
        scripted_fn_sub = torch.jit.script(test_sub, example_inputs=[(3.9, 4.10)])
        self.assertEqual(scripted_fn_sub(6.5, 2.9), test_sub(6.5, 2.9))

        def test_mul(a, b):
            return a * b

        make_global(test_mul)
        scripted_fn_mul = torch.jit.script(test_mul, example_inputs=[(-10, 9)])
        self.assertEqual(scripted_fn_mul(-1, 3), test_mul(-1, 3))

        def test_args_complex(real, img):
            return torch.complex(real, img)

        make_global(test_args_complex)
        scripted_fn_complex = torch.jit.script(
            test_args_complex, example_inputs=[(torch.rand(3, 4), torch.rand(3, 4))]
        )
        arg1, arg2 = torch.rand(3, 4), torch.rand(3, 4)
        self.assertEqual(scripted_fn_complex(arg1, arg2), test_args_complex(arg1, arg2))

        def test_bool(a):
            if a:
                return -1
            else:
                return 0

        make_global(test_bool)
        scripted_fn_bool = torch.jit.script(test_bool, example_inputs=[(True,)])
        self.assertEqual(scripted_fn_bool(True), test_bool(True))

        def test_str(a):
            if a == "":
                return False
            else:
                return True

        make_global(test_str)
        scripted_fn_str = torch.jit.script(test_str, example_inputs=[("",)])
        self.assertEqual(scripted_fn_str("abc"), test_str("abc"))

    def test_pdt_list_and_tuple(self):
        def test_list_and_tuple(a):
            return sum(a)

        make_global(test_list_and_tuple)

        scripted_fn_float_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([4.9, 8.9],)]
        )
        self.assertEqual(
            scripted_fn_float_list_input([11.9, 7.6]), test_list_and_tuple([11.9, 7.6])
        )

        scripted_fn_bool_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([True, False, True],)]
        )
        self.assertEqual(
            scripted_fn_bool_list_input([True, True, True]),
            test_list_and_tuple([True, True, True]),
        )

        scripted_fn_int_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([3, 4, 5],)]
        )
        self.assertEqual(
            scripted_fn_int_list_input([1, 2, 3]), test_list_and_tuple([1, 2, 3])
        )

        scripted_fn_float_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((4.9, 8.9),)]
        )
        self.assertEqual(
            scripted_fn_float_tuple_input((11.9, 7.6)), test_list_and_tuple((11.9, 7.6))
        )

        scripted_fn_bool_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((True, False, True),)]
        )
        self.assertEqual(
            scripted_fn_bool_tuple_input((True, True, True)),
            test_list_and_tuple((True, True, True)),
        )

        scripted_fn_int_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((3, 4, 5),)]
        )
        self.assertEqual(
            scripted_fn_int_tuple_input((1, 2, 3)), test_list_and_tuple((1, 2, 3))
        )

    def test_nested_list_and_tuple(self):
        def test_nested_list(inp):
            return [sum(v) for v in inp]

        def test_nested_tuple(inp):
            ans = 0.0
            for tup in inp:
                for val in tup:
                    if val > 0:
                        ans *= val
            return ans

        make_global(test_nested_list, test_nested_tuple)

        list_inp = [
            [
                1,
                2,
                3,
            ],
            [
                5,
                6,
                7,
            ],
        ]
        scripted_fn = torch.jit.script(
            test_nested_list,
            example_inputs=[
                (list_inp,),
            ],
        )
        inp = [
            [
                0,
                4,
                7,
            ],
            [
                8,
                11,
            ],
            [
                6,
                -1,
                -20,
            ],
        ]
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_list(
                inp,
            ),
        )

        list_inp = (
            [
                1,
                2,
                3,
            ],
            [
                5,
                6,
                7,
            ],
        )
        scripted_fn = torch.jit.script(
            test_nested_list,
            example_inputs=[
                (list_inp,),
            ],
        )
        inp = (
            [
                0,
                4,
                7,
            ],
            [
                8,
                11,
            ],
            [
                6,
                -1,
                -20,
            ],
        )
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_list(
                inp,
            ),
        )

        tup_inp = [
            (
                1.0,
                2.6,
                3.7,
            ),
            (
                5.7,
                6.1,
                1.7,
            ),
        ]
        scripted_fn = torch.jit.script(
            test_nested_tuple,
            example_inputs=[
                (tup_inp,),
            ],
        )
        inp = [
            (
                1.0,
                4.1,
                7.4,
            ),
            (
                4.8,
                1.1,
                -1.2,
            ),
            (
                6.3,
                -1.3,
                -2.0,
            ),
        ]
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_tuple(
                inp,
            ),
        )

        tup_inp = (
            (
                True,
                False,
                True,
            ),
            (
                False,
                False,
                False,
            ),
        )
        scripted_fn = torch.jit.script(
            test_nested_tuple,
            example_inputs=[
                (tup_inp,),
            ],
        )
        inp = (
            (
                True,
                True,
                True,
            ),
            (
                False,
                False,
                True,
            ),
        )
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_tuple(
                inp,
            ),
        )

    def test_pdt_dict(self):
        def test_dict(a):
            return a["foo"]

        def test_dict_int_list(a):
            return a[1]

        make_global(test_dict, test_dict_int_list)

        str_bool_inp = {"foo": True, "bar": False}
        scripted_fn = torch.jit.script(test_dict, example_inputs=[(str_bool_inp,)])
        self.assertEqual(
            scripted_fn(
                {"foo": False, "bar": True},
            ),
            test_dict(
                {"foo": False, "bar": True},
            ),
        )

        str_list_inp = {0: [True, False], 1: [False, True]}
        scripted_fn = torch.jit.script(
            test_dict_int_list, example_inputs=[(str_list_inp,)]
        )
        self.assertEqual(
            scripted_fn(
                {0: [False, False], 1: [True, True]},
            ),
            test_dict_int_list(
                {0: [False, False], 1: [True, True]},
            ),
        )

    def test_any(self):
        def test_multiple_types(a):
            assert not isinstance(a, bool)
            return a

        def test_multiple_type_refinement(a):
            if isinstance(a, bool):
                return 1
            elif isinstance(a, int):
                return 1 + a
            elif isinstance(a, float):
                return 1 + int(a)
            else:
                return -1

        make_global(test_multiple_types, test_multiple_type_refinement)

        scripted_fn = torch.jit.script(
            test_multiple_types, example_inputs=[(1,), ("abc",), (8.9,), ([3, 4, 5],)]
        )
        self.assertEqual(scripted_fn(10), test_multiple_types(10))
        self.assertEqual(scripted_fn("def"), test_multiple_types("def"))
        self.assertEqual(scripted_fn(7.89999), test_multiple_types(7.89999))
        self.assertEqual(scripted_fn([10, 11, 14]), test_multiple_types([10, 11, 14]))

        scripted_fn = torch.jit.script(
            test_multiple_type_refinement,
            example_inputs=[
                (1,),
                ("abc",),
                (8.9,),
                ([3, 4, 5],),
                (True,),
                ({"a": True},),
            ],
        )
        self.assertEqual(scripted_fn(10), test_multiple_type_refinement(10))
        self.assertEqual(scripted_fn("def"), test_multiple_type_refinement("def"))
        self.assertEqual(scripted_fn(7.89999), test_multiple_type_refinement(7.89999))
        self.assertEqual(
            scripted_fn([10, 11, 14]), test_multiple_type_refinement([10, 11, 14])
        )
        self.assertEqual(scripted_fn(False), test_multiple_type_refinement(False))
        self.assertEqual(
            scripted_fn({"abc": True, "def": False}),
            test_multiple_type_refinement({"abc": True, "def": False}),
        )

    def test_class_as_profiled_types(self):
        class UserDefinedClass:
            def fn(self, b) -> Any:
                assert b is not None
                if isinstance(b, int):
                    return b if b > 0 else -1
                elif isinstance(b, float):
                    return b if b > 0.0 else -1.0
                return 0

        def test_model(a, m):
            assert not isinstance(a, bool)
            return m.fn(a)

        make_global(UserDefinedClass, test_model)

        user_class = UserDefinedClass()
        scripted_fn = torch.jit.script(
            test_model,
            example_inputs=[
                (
                    10,
                    user_class,
                ),
                (
                    10.9,
                    user_class,
                ),
            ],
        )
        self.assertEqual(
            scripted_fn(
                100,
                user_class,
            ),
            test_model(100, user_class),
        )
        self.assertEqual(
            scripted_fn(
                1.9,
                user_class,
            ),
            test_model(1.9, user_class),
        )

    def test_class_with_args_as_profiled_types(self):
        class ClassWithArgs:
            def __init__(self, a: bool):
                self.a = a

            def fn(self, b):
                if self.a:
                    return b
                else:
                    return -1

        def test_model_with_args(a, m):
            assert not isinstance(a, bool)
            return m.fn(a)

        make_global(ClassWithArgs, test_model_with_args)

        user_class = ClassWithArgs(False)
        scripted_fn = torch.jit.script(
            test_model_with_args,
            example_inputs=[
                (
                    10,
                    user_class,
                ),
                (
                    10.9,
                    user_class,
                ),
            ],
        )
        self.assertEqual(
            scripted_fn(
                100,
                ClassWithArgs(True),
            ),
            test_model_with_args(100, ClassWithArgs(True)),
        )

    def test_nn_parameter_as_arg(self):
        class TestNNParameter(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inp = torch.nn.Parameter(torch.ones(2, 3))

            def add_nn_parameter_with_int(self, x, y):
                return torch.add(x, y)

            def forward(self, y):
                return self.add_nn_parameter_with_int(self.inp, y)

        make_global(TestNNParameter)
        pdt_model = TestNNParameter()
        scripted_fn = torch.jit.script(
            pdt_model,
            example_inputs={
                pdt_model: [
                    (10,),
                ],
            },
        )
        self.assertEqual(scripted_fn(20), pdt_model(20))

    def test_fx_tracing_with_typing(self):
        class FXModelOutput(NamedTuple):
            result: List[int]

        class FXModel(torch.nn.Module):
            def forward(self, a) -> FXModelOutput:
                result = FXModelOutput(result=a)
                return result

        make_global(FXModel, FXModelOutput)
        pdt_model = FXModel()
        scripted_fn = torch.jit.script(
            pdt_model,
            example_inputs={
                pdt_model: [
                    (
                        [
                            10,
                            20,
                        ],
                    ),
                ],
            },
        )
        self.assertEqual(scripted_fn([20]), pdt_model([20]))

    def test_nonetype_as_optional_of_type(self):
        def test_none(a) -> Any:
            if a is None:
                return 0
            else:
                return a + torch.ones(1)

        make_global(test_none)

        scripted_fn = torch.jit.script(test_none, example_inputs=[(None,), (10.6,)])
        self.assertEqual(
            scripted_fn(
                30.9,
            ),
            test_none(
                30.9,
            ),
        )

        scripted_fn = torch.jit.script(test_none, example_inputs=[(None,), (10,)])
        self.assertEqual(
            scripted_fn(
                2,
            ),
            test_none(
                2,
            ),
        )

        scripted_fn = torch.jit.script(
            test_none, example_inputs=[(None,), (torch.Tensor(1),)]
        )
        self.assertEqual(
            scripted_fn(
                torch.ones(1),
            ),
            test_none(
                torch.ones(1),
            ),
        )
