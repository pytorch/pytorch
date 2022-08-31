# Owner(s): ["oncall: jit"]

import os
import sys

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestTensorCreationOps(JitTestCase):
    """
    A suite of tests for ops that create tensors.
    """

    def test_randperm_default_dtype(self):
        def randperm(x: int):
            perm = torch.randperm(x)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert perm.dtype == torch.int64

        self.checkScript(randperm, (3, ))

    def test_randperm_specifed_dtype(self):
        def randperm(x: int):
            perm = torch.randperm(x, dtype=torch.float)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert perm.dtype == torch.float

        self.checkScript(randperm, (3, ))

    def test_triu_indices_default_dtype(self):
        def triu_indices(rows: int, cols: int):
            indices = torch.triu_indices(rows, cols)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int64

        self.checkScript(triu_indices, (3, 3))

    def test_triu_indices_specified_dtype(self):
        def triu_indices(rows: int, cols: int):
            indices = torch.triu_indices(rows, cols, dtype=torch.int32)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int32

        self.checkScript(triu_indices, (3, 3))

    def test_tril_indices_default_dtype(self):
        def tril_indices(rows: int, cols: int):
            indices = torch.tril_indices(rows, cols)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int64

        self.checkScript(tril_indices, (3, 3))

    def test_tril_indices_specified_dtype(self):
        def tril_indices(rows: int, cols: int):
            indices = torch.tril_indices(rows, cols, dtype=torch.int32)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int32

        self.checkScript(tril_indices, (3, 3))
