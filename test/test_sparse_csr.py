import torch

# NOTE: These tests are inspired from test_sparse.py and may duplicate some behaviour.
# Need to think about merging them both sometime down the line.

# Major differences between testing of CSR and COO is that we don't need to test CSR
# for coalesced/uncoalesced behaviour.

# TODO: remove this global setting
# Sparse tests use double as the default dtype
torch.set_default_dtype(torch.double)

import itertools
import functools
import random
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, load_tests

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestSparseGCS(TestCase):
    def setUp(self):
        # These parameters control the various ways we can run the test.
        # We will subclass and override this method to implement CUDA
        # tests
        self.is_cuda = False
        self.is_uncoalesced = False
        self.device = 'cpu'
        self.exact_dtype = True
        self.value_dtype = torch.float64
        self.index_tensor = lambda *args: torch.tensor(*args, dtype=torch.int64, device=self.device)
        self.value_empty = lambda *args: torch.empty(*args, dtype=self.value_dtype, device=self.device)
        self.value_tensor = lambda *args: torch.tensor(*args, dtype=self.value_dtype, device=self.device)

        def sparse_tensor_factory(*args, **kwargs):
            kwargs['dtype'] = kwargs.get('dtype', self.value_dtype)
            kwargs['device'] = kwargs.get('device', self.device)
            return torch.sparse_gcs_tensor(*args, **kwargs)
            
        self.sparse_tensor = sparse_tensor_factory
        super(TestSparseGCS, self).setUp()
    
    def test_gcs_layout(self):
        self.assertEqual(str(torch.sparse_gcs), 'torch.sparse_gcs')
        self.assertEqual(type(torch.sparse_gcs), torch.layout)

    def test_sparse_gcs_from_dense(self):
        sp = torch.tensor([[1, 2], [3, 4]]).to_sparse_gcs(torch.empty(0,1), 1)
        
        print(sp.pointers)
        print(sp.indices)
        print(sp.values)
        print(sp.fill_value)
    
    def test_sparse_gcs_constructor(self):
        pass

if __name__ == '__main__':
    run_tests()
