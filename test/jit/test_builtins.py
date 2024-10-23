# Owner(s): ["oncall: jit"]

import inspect
import os
import sys
import unittest
from typing import Dict, List

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestBuiltins(JitTestCase):
    """
    Tests for TorchScript support of Python builtin functions.
    """

    def test_has_attr(self):
        class HasA(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 0

        class HasB(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = 1

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mods = torch.nn.ModuleList([HasA(), HasB()])

            def forward(self):
                # use a list to encode hasattr results
                l = torch.jit.annotate(List[int], [])
                for mod in self.mods:
                    l.append(int(hasattr(mod, "a")))
                    l.append(int(hasattr(mod, "b")))
                    # actually retrieve the attr to test static refinement
                    if hasattr(mod, "a"):
                        l.append(mod.a)
                    if hasattr(mod, "b"):
                        l.append(mod.b)
                return l

        self.checkModule(Mod(), ())

    def test_has_attr_invalid_args(self):
        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod = torch.nn.Linear(1, 1)

            def forward(self, name):
                # not allowed, `name` must be static.
                return hasattr(self.mod, name)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())

        class Mod(torch.nn.Module):
            def forward(self, name):
                # not allowed, `torch.rand` is not a class type
                return hasattr(torch.rand(2, 3), name)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())

    def test_del(self):
        def fn(x: List[int]) -> List[int]:
            a = x * 2
            del a
            return x

        self.checkScript(fn, ([1, 2, 3],))

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @torch.jit.script
            def fn(x):
                a = x**2
                del a
                return a  # noqa: F821

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @torch.jit.script
            def fn(x):
                a = x**2
                if a:
                    del a
                return a

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "b"):

            @torch.jit.script
            def fn(x):
                a = x**2
                del b  # noqa: F821
                return a

    def test_del_multiple_operands(self):
        def fn(x: List[int]) -> List[int]:
            a, b, c = x[0], x[1], x[2]
            del a, b, c
            return x

        self.checkScript(fn, ([1, 2, 3],))

        def del_list_multiple_operands(x: List[int]) -> List[int]:
            del x[0], x[1]
            return x

        py_out = del_list_multiple_operands([0, 1, 2])
        jit_out = torch.jit.script(del_list_multiple_operands)([0, 1, 2])
        self.assertEqual(py_out, jit_out)

        def del_dict_multiple_operands(x: Dict[str, int]) -> Dict[str, int]:
            del x["hi"], x["there"]
            return x

        py_out = del_dict_multiple_operands({"hi": 5, "there": 6})
        jit_out = torch.jit.script(del_dict_multiple_operands)({"hi": 5, "there": 6})
        self.assertEqual(py_out, jit_out)


class TestTensorBuiltins(JitTestCase):
    def test_tensor_properties(self):
        def should_keep(tensor, name):
            if inspect.isroutine(getattr(tensor, name)):
                return False
            if name.startswith("_"):
                return False
            return True

        tensor = torch.arange(4, dtype=torch.float).view(2, 2)
        keys = dir(tensor)

        # real and imag are only implemented for complex tensors.
        self.assertRaises(RuntimeError, lambda: should_keep(tensor, "imag"))
        keys.remove("imag")

        properties = [p for p in keys if should_keep(tensor, p)]

        code_template = """
        def fn(x):
            return x.{}
        """

        EQUALITY_MISMATCH = {
            # TorchScript doesn't have real enums so they return an int instead
            # of the actual value
            "dtype",
            "layout",
        }
        MISSING_PROPERTIES = {
            "grad_fn",
            # This is an undocumented property so it's not included
            "output_nr",
            # This has a longer implementation, maybe not worth copying to
            # TorchScript if named tensors don't work there anyways
            "names",
        }

        for p in properties:
            if p in MISSING_PROPERTIES:
                continue
            code = code_template.format(p)
            cu = torch.jit.CompilationUnit()
            cu.define(code)
            if p in EQUALITY_MISMATCH:
                continue
            self.assertEqual(getattr(tensor, p), cu.fn(tensor))

    def test_tensor_subscript_assign(self):
        def fn1(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = torch.tensor(2, dtype=torch.uint8)
            return a

        def fn2(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = 2
            return a

        def fn3(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = 2
            return a

        def fn4(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = torch.tensor(2, dtype=torch.uint8)
            return a

        def fn5(x):
            a = torch.zeros_like(x, dtype=torch.float32)
            a[torch.tensor(0)] = 2
            return a

        for fn in (fn1, fn2, fn3, fn4, fn5):
            self.checkScript(fn, (torch.zeros(2, dtype=torch.uint8),))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_tensor_subscript_assign_device(self):
        def fn6(x):
            a = torch.zeros_like(x, dtype=torch.float32, device="cuda")
            a[torch.tensor(0)] = 2
            return a

        self.checkScript(fn6, (torch.zeros(2, dtype=torch.float32, device="cuda"),))

    def test_tensor_item(self):
        def test_scalar_cast(x):
            scalar = x.item()
            return int(scalar), float(scalar)

        graph = torch.jit.script(test_scalar_cast).graph
        FileCheck().check("(int, float) = prim::TupleConstruct").run(graph)
        self.checkScript(test_scalar_cast, (torch.tensor(1.0),))
        self.checkScript(test_scalar_cast, (torch.tensor(1),))

    def test_method_on_number(self):
        def func():
            c = 1
            return c.add(1)

        with self.assertRaisesRegex(RuntimeError, "object has no attribute or method"):
            torch.jit.script(func)

    # testing implicit conversion of tensors to scalars to match function arguments
    def test_scalar_to_num_conversions(self):
        @torch.jit.script
        def multiple_defs(x):
            c = 1
            x = x + c
            return x

        self.assertTrue("ImplicitTensorToNum" not in str(multiple_defs.graph))

        @torch.jit.script
        def tensor_to_int_script(x, tensor):
            return x.unsqueeze(tensor)

        # location present in error message
        with self.assertRaisesRegex(RuntimeError, "x.unsqueeze"):
            tensor_to_int_script(torch.tensor([2]), torch.tensor([2, 2]))

        def tensor_to_int(x, tensor):
            return x.unsqueeze(tensor)

        @torch.jit.script
        def tensor_to_float_script(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        def tensor_to_float(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        x = torch.zeros(10)
        # float tensor, float tensor with grad, int tensor (can't set grad on int tensor)
        tensors = [
            torch.tensor(1.1),
            torch.tensor(1.1, requires_grad=True),
            torch.tensor(0),
            torch.tensor([2]),
        ]

        script_funs = [tensor_to_int_script, tensor_to_float_script]
        funs = [tensor_to_int, tensor_to_float]

        # return the result, or whether exception was thrown
        def test_func(func, x, tensor):
            try:
                result = func(x, tensor)
            except RuntimeError as e:
                result = True
            except TypeError as e:
                result = True
            return result

        # assert result or exception equal for each (function, inputs)
        for tensor in tensors:
            for i in range(len(script_funs)):
                self.assertEqual(
                    test_func(script_funs[i], x, tensor), test_func(funs[i], x, tensor)
                )
