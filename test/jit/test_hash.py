# Owner(s): ["oncall: jit"]

import os
import sys

import torch

from typing import Tuple, List

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestHash(JitTestCase):
    def test_hash_tuple(self):
        def fn(t1: Tuple[int, int], t2: Tuple[int, int]) -> bool:
            return hash(t1) == hash(t2)

        self.checkScript(fn, ((1, 2), (1, 2)))
        self.checkScript(fn, ((1, 2), (3, 4)))
        self.checkScript(fn, ((1, 2), (2, 1)))

    def test_hash_tuple_nested_unhashable_type(self):
        # Tuples may contain unhashable types like `list`, check that we error
        # properly in that case.
        @torch.jit.script
        def fn_unhashable(t1: Tuple[int, List[int]]):
            return hash(t1)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "unhashable", "hash"):
            fn_unhashable((1, [1]))

    def test_hash_tensor(self):
        """Tensors should hash by identity"""
        def fn(t1, t2):
            return hash(t1) == hash(t2)

        tensor1 = torch.tensor(1)
        tensor1_clone = torch.tensor(1)
        tensor2 = torch.tensor(2)

        self.checkScript(fn, (tensor1, tensor1))
        self.checkScript(fn, (tensor1, tensor1_clone))
        self.checkScript(fn, (tensor1, tensor2))

    def test_hash_none(self):
        def fn():
            n1 = None
            n2 = None
            return hash(n1) == hash(n2)

        self.checkScript(fn, ())

    def test_hash_bool(self):
        def fn(b1: bool, b2: bool):
            return hash(b1) == hash(b2)

        self.checkScript(fn, (True, False))
        self.checkScript(fn, (True, True))
        self.checkScript(fn, (False, True))
        self.checkScript(fn, (False, False))

    def test_hash_float(self):
        def fn(f1: float, f2: float):
            return hash(f1) == hash(f2)

        self.checkScript(fn, (1.2345, 1.2345))
        self.checkScript(fn, (1.2345, 6.789))
        self.checkScript(fn, (1.2345, float("inf")))
        self.checkScript(fn, (float("inf"), float("inf")))
        self.checkScript(fn, (1.2345, float('nan')))
        if sys.version_info < (3, 10):
            # Hash of two nans are not guaranteed to be equal. From https://docs.python.org/3/whatsnew/3.10.html :
            # Hashes of NaN values of both float type and decimal.Decimal type now depend on object identity.
            self.checkScript(fn, (float("nan"), float("nan")))
        self.checkScript(fn, (float("nan"), float("inf")))

    def test_hash_int(self):
        def fn(i1: int, i2: int):
            return hash(i1) == hash(i2)

        self.checkScript(fn, (123, 456))
        self.checkScript(fn, (123, 123))
        self.checkScript(fn, (123, -123))
        self.checkScript(fn, (-123, -123))
        self.checkScript(fn, (123, 0))

    def test_hash_string(self):
        def fn(s1: str, s2: str):
            return hash(s1) == hash(s2)

        self.checkScript(fn, ("foo", "foo"))
        self.checkScript(fn, ("foo", "bar"))
        self.checkScript(fn, ("foo", ""))

    def test_hash_device(self):
        def fn(d1: torch.device, d2: torch.device):
            return hash(d1) == hash(d2)

        gpu0 = torch.device('cuda:0')
        gpu1 = torch.device('cuda:1')
        cpu = torch.device('cpu')
        self.checkScript(fn, (gpu0, gpu0))
        self.checkScript(fn, (gpu0, gpu1))
        self.checkScript(fn, (gpu0, cpu))
        self.checkScript(fn, (cpu, cpu))
