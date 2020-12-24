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
