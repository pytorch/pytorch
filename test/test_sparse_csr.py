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

    def test_sparse_coo_const(self):
        torch.sparse_coo_tensor(
            torch.LongTensor([[0], [1], [2]]).transpose(1, 0).clone().detach(),
            torch.FloatTensor([3, 4, 5]),
            torch.Size([3]),
            device=self.device)

    def test_sparse_gcs_from_dense(self):
        def make_sparse_gcs(data, reduction=None, fill_value=-1):
            import itertools
            from collections import defaultdict

            def get_shape(data):
                if isinstance(data, (list, tuple)):
                    dims = len(data)
                    if dims == 0:
                        return (0,)
                    return (dims, ) + get_shape(data[0])
                return ()

            def make_strides(shape, dims=None):
                if dims is None:
                    dims = tuple(range(len(shape)))
                ndims = len(dims)
                if ndims == 0:
                    return ()
                strides = [1]
                for i in range(ndims - 1):
                    strides.insert(0, strides[0] * shape[dims[ndims - i - 1]])
                return tuple(strides)

            def apply_reduction(index, strides, dims):
                return sum(strides[k] * index[dims[k]] for k in range(len(dims)))
            
            shape = get_shape(data)
            N = len(shape)
            # TODO: N=0, N=1
            if reduction is None:
                dims1 = tuple(range(N//2))
                dims2 = tuple(range(N//2, N))
                reduction = dims1 + dims2 + (N//2,)
            else:
                l = reduction[-1]
                dims1 = reduction[:l]
                dims2 = reduction[l:-1]

            strides1 = make_strides(dims1)
            strides2 = make_strides(dims2)
            print(f'{shape} {strides1} {strides2} {dims1} {dims2}')
            # <row>: <list of (colindex, value)>
            col_value = defaultdict(list)
            for index in itertools.product(*map(range, shape)):
                v = data
                for i in index:
                    v = v[i]
                if v == fill_value:
                    continue
                p1 = apply_reduction(index, strides1, dims1)
                p2 = apply_reduction(index, strides2, dims2)
                col_value[p1].append((p2, v))
            ro = [0]
            co = []
            values = []
            for i in range(max(col_value)+1):
                cv = col_value.get(i, [])
                ro.append(ro[-1] + len(cv))
                cv.sort()
                c, v = zip(*cv)
                co.extend(c)
                values.extend(v)

            print(f"{torch.tensor(ro)} {torch.tensor(co)} {torch.tensor(values)} {torch.tensor(reduction)} {shape}")
            return torch.sparse_gcs_tensor(torch.tensor(ro), torch.tensor(co), torch.tensor(values),
                                           torch.tensor(reduction), shape, fill_value)
        
        sp = make_sparse_gcs([[1, 2], [3, 4]])
        
        print(sp.pointers())
        print(sp.indices())
        print(sp.values())
    
    def test_sparse_gcs_constructor(self):
        pass

if __name__ == '__main__':
    run_tests()
