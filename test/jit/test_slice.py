import os
import sys

import torch
from typing import List

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Tests that Python slice class is supported in TorchScript
class TestSlice(JitTestCase):
    def test_slice_kwarg(self):
        def slice_kwarg(x: List[int]):
            return x[slice(1, stop=2)]

        with self.assertRaisesRegex(RuntimeError, "Slice does not accept any keyword arguments"):
            torch.jit.script(slice_kwarg)

    def test_slice_three_nones(self):
        def three_nones(x: List[int]):
            return x[slice(None, None, None)]

        self.checkScript(three_nones, (range(10),))

    def test_slice_two_nones(self):
        def two_nones(x: List[int]):
            return x[slice(None, None)]

        self.checkScript(two_nones, (range(10),))

    def test_slice_one_none(self):
        def one_none(x: List[int]):
            return x[slice(None)]

        self.checkScript(one_none, (range(10),))

    def test_slice_stop_only(self):
        def fn(x: List[int]):
            return x[slice(5)]
        self.checkScript(fn, (range(10),))

    def test_slice_stop_only_with_nones(self):
        def fn(x: List[int]):
            return x[slice(None, 5, None)]
        self.checkScript(fn, (range(10),))

    def test_slice_start_stop(self):
        def fn(x: List[int]):
            return x[slice(1, 5)]

        self.checkScript(fn, (range(10),))

    def test_slice_start_stop_with_none(self):
        def fn(x: List[int]):
            return x[slice(1, 5, None)]

        self.checkScript(fn, (range(10),))

    def test_slice_start_stop_step(self):
        def fn(x: List[int]):
            return x[slice(0, 6, 2)]

        self.checkScript(fn, (range(10),))

    def test_slice_string(self):
        def fn(x: str):
            return x[slice(None, 3, 1)]

        self.checkScript(fn, ("foo_bar",))

    def test_slice_tensor(self):
        def fn(x: torch.Tensor):
            return x[slice(None, 3, 1)]

        self.checkScript(fn, (torch.ones(10),))

    def test_slice_tensor_multidim(self):
        def fn(x: torch.Tensor):
            return x[slice(None, 3, 1), 0]

        self.checkScript(fn, (torch.ones((10, 10)),))

    def test_slice_tensor_multidim_with_dots(self):
        def fn(x: torch.Tensor):
            return x[slice(None, 3, 1), ...]

        self.checkScript(fn, (torch.ones((10, 10)),))

    def test_slice_as_variable(self):
        def fn(x: List[int]):
            a = slice(1)
            return x[a]

        self.checkScript(fn, (range(10),))

    def test_slice_stop_clipped(self):
        def fn(x: List[int]):
            return x[slice(1000)]

        self.checkScript(fn, (range(10),))

    def test_slice_dynamic_index(self):
        def t(x):
            slice1 = x[0:1]
            zero = 0
            one = zero + 1
            slice2 = x[zero:one]
            return slice1 + slice2

        self.checkScript(t, (torch.zeros(3, 2, 3),))

    def test_tuple_slicing(self):
        def tuple_slice(a):
            if bool(a):
                b = (1, 2, 3, 4)
            else:
                b = (4, 3, 2, 1)
            c = b[-4:4]
            e = c[1:-1]
            return e

        self.checkScript(tuple_slice, (torch.tensor([1]),), optimize=True)
        scripted_fn = torch.jit.script(tuple_slice)
        self.assertEqual(scripted_fn(torch.tensor(1)), (2, 3))
        tuple_graph = scripted_fn.graph
        slices = tuple_graph.findAllNodes("prim::TupleConstruct")
        num_outputs = set(len(x.output().type().elements()) for x in slices)
        # there should be only one tupleSlice with length of 2
        self.assertTrue(num_outputs == {2})
        self.run_pass('lower_all_tuples', tuple_graph)
        self.assertTrue('Tuple' not in str(tuple_graph))

    def test_module_list_slicing(self):
        class Bar(torch.nn.Module):
            def __init__(self, identifier: str):
                super().__init__()
                self.identifier = identifier

            def forward(self):
                return 0

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                module_list = [Bar("A"), Bar("B"), Bar("C"), Bar("D"), Bar("E")]
                self.test = torch.nn.ModuleList(module_list)

            def forward(self):
                return self.test[::-2], self.test[1:4:]

        scripted_foo = torch.jit.script(Foo())
        result1, result2 = scripted_foo()

        self.assertEqual(len(result1), 3)
        self.assertEqual(result1[0].identifier, "E")
        self.assertEqual(result1[1].identifier, "C")
        self.assertEqual(result1[2].identifier, "A")

        self.assertEqual(len(result2), 3)
        self.assertEqual(result2[0].identifier, "B")
        self.assertEqual(result2[1].identifier, "C")
        self.assertEqual(result2[2].identifier, "D")
