import torch
from torch import sparse
from torch._utils import _import_dotted_name # ick

import itertools
import random
import unittest
from common import TestCase, run_tests
from common_nn import TEST_CUDA
from numbers import Number


def cpu_only(inner):
    def outer(self, *args, **kwargs):
        if self.is_cuda:
            raise unittest.SkipTest("Test is CPU-only")
        inner(self, *args, **kwargs)
    return outer


class SparseSize:
    """Represents the size of a sparse tensor.  There is a logical size which is
    just a torch.Size (what you would have gotten if you called toDense) but
    sparse tensors also carry extra metadata, which this class records."""
    # TODO: Maybe promote this into a real thing.
    def __init__(self, size, dimI, nnz):
        self._size = size
        self._nnz = nnz
        self._dimI = dimI
    @staticmethod
    def from_sparse(s):
        return SparseSize(s.size(), s._indices().size()[0], s._nnz())
    def size(self):
        """Logical size of the sparse tensor; i.e., what you would get if
        you called s.to_dense().size()"""
        return self._size
    def setSize(self, size):
        """Change the logical size of the sparse tensor."""
        assert len(size) >= self._dimI
        self._size = size
        return self
    def dimI(self):
        """Number of sparse dimensions in the sparse tensor.  Despite this
        name, this is NOT the dimension of indices (indices is always 2D)."""
        return self._dimI
    def dimV(self):
        """Number of dense dimensions in the sparse tensor.  Despite this name,
        this value is ONE LESS than the dimensionality of _values()."""
        return len(self.size()) - self.dimI()
    def nnz(self):
        """Number of non-zero values in the sparse tensor"""
        return self._nnz
    def indicesSize(self):
        """Size of _indices()"""
        return torch.Size([self.dimI(), self.nnz()])
    def valuesSize(self):
        """Size of _values()"""
        return torch.Size((self.nnz(),) + self.size()[self.dimI():])
    def new_sparse(self, new_type, indices=None, values=None):
        """Create a new sparse tensor with the size specified by this size."""
        # TODO: This code is disgusting
        mod = new_type.__module__
        cls = new_type.__name__
        if indices is None:
            indices = mod.LongTensor(self.indicesSize())
        else:
            assert indices.size() == self.indicesSize()
        if values is None:
            value_type = _import_dotted_name(mod.replace('.sparse', '') + '.' + cls)
            values = value_type(self.valuesSize())
        else:
            assert values.size() == self.valuesSize()
        return new_type(indices, values, self.size())

def dense2sparse(dense_type):
    mod = dense_type.__module__
    cls = dense_type.__name__
    return _import_dotted_name(mod + ".sparse." + cls)

class TestSparse(TestCase):

    def setUp(self):
        # These parameters control the various ways we can run the test.
        # We will subclass and override this method to implement CUDA
        # tests
        self.is_cuda = False
        self.is_uncoalesced = False
        self.IndexTensor = torch.LongTensor
        self.ValueTensor = torch.DoubleTensor
        self.SparseTensor = torch.sparse.DoubleTensor

    def _gen_sparse(self, d, nnz, with_size):
        # TODO: Consider implementing this in the CUDA case by directly
        # performing the operations on the GPU.  You won't be able to
        # use torch.rand/torch.randn in this case because they are
        # CPU-only.  If you do this, you can remove the is_cuda branch
        # at the end.
        #
        # If you do this, be sure to update assert_uncoalesced too

        if isinstance(with_size, Number):
            with_size = [with_size] * d

        if self.is_uncoalesced:
            # We want to generate a tensor with a lot of uncoalesced
            # entries to stress test whether or not we handle this
            # (subtle) case correctly
            v_size = [nnz * 2] + list(with_size[d:])
            v = torch.randn(*v_size)
            r = torch.rand(d, nnz)
            # Repeat the indexes, so every position shows up twice
            i = torch.cat([r, r], dim=1) * \
                torch.Tensor(with_size[:d]).repeat(nnz * 2, 1).transpose(0, 1)
            i = i.type(torch.LongTensor)
            x = torch.sparse.DoubleTensor(i, v, torch.Size(with_size))
            self.assert_uncoalesced(x)
        else:
            # Generate a sparse tensor with d sparse dimensions; the
            # rest the dimensions with_size[d:] are dense.
            v_size = [nnz] + list(with_size[d:])
            v = torch.randn(*v_size)
            i = torch.rand(d, nnz) * \
                torch.Tensor(with_size[:d]).repeat(nnz, 1).transpose(0, 1)
            i = i.type(torch.LongTensor)
            x = torch.sparse.DoubleTensor(i, v, torch.Size(with_size))

        if self.is_cuda:
            return x.cuda(), i.cuda(), v.cuda()
        else:
            return x, i.clone(), v.clone()

    def assert_uncoalesced(self, x):
        """
        Test if a CPU tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        assert not x.is_coalesced()
        # Strategy: construct a new sparse tensor with the raw value
        # field overwritten to a tensor of ones, coalesce it, and then
        # check if any value entries are > 1 (which indicates that the
        # original was uncoalesced.)
        i = x._indices().clone()
        v = x._values().clone().fill_(1)
        y = torch.sparse.DoubleTensor(i, v, x.size())
        z = self.safeCoalesce(y)
        assert (z._values() > 1).sum() > 0

    def randn(self, *args, **kwargs):
        """
        Variant of torch.randn that also works in the TEST_CUDA case.
        """
        # TODO: Put this in torch.cuda.randn
        return self.ValueTensor(*args, **kwargs).normal_()

    def test_basic(self):
        x, i, v = self._gen_sparse(3, 10, 100)

        self.assertEqual(i, x._indices())
        self.assertEqual(v, x._values())

        x, i, v = self._gen_sparse(3, 10, [100, 100, 100])
        self.assertEqual(i, x._indices())
        self.assertEqual(v, x._values())
        self.assertEqual(x.ndimension(), 3)
        self.assertEqual(x.coalesce()._nnz(), 10)
        for i in range(3):
            self.assertEqual(x.size(i), 100)

        # Make sure we can access empty indices / values
        x = self.SparseTensor()
        self.assertEqual(x._indices().numel(), 0)
        self.assertEqual(x._values().numel(), 0)

    def test_to_dense(self):
        i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        v = self.ValueTensor([2, 1, 3, 4])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5]))
        res = self.ValueTensor([
            [[2, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 3, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 4]],
        ])

        x.to_dense()  # Tests double to_dense for memory corruption
        x.to_dense()
        x.to_dense()
        self.assertEqual(res, x.to_dense())

    def test_to_dense_hybrid(self):
        i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ])
        v = self.ValueTensor([[2, 3], [1, 2], [3, 4], [4, 5]])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 2]))
        res = self.ValueTensor([
            [[2, 3],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[1, 2],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[3, 4],
             [0, 0],
             [0, 0],
             [4, 5]],
        ])

        x.to_dense()  # Tests double to_dense for memory corruption
        x.to_dense()
        x.to_dense()
        self.assertEqual(res, x.to_dense())

    def test_contig(self):
        i = self.IndexTensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = self.ValueTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x = self.SparseTensor(i, v, torch.Size([100, 100]))
        exp_i = self.IndexTensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = self.ValueTensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7])
        x = self.safeCoalesce(x)
        self.assertEqual(exp_i, x._indices())
        self.assertEqual(exp_v, x._values())

        i = self.IndexTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.ValueTensor([3, 2, 4, 1])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.ValueTensor([2, 1, 3, 4])

        x = self.safeCoalesce(x)
        self.assertEqual(exp_i, x._indices())
        self.assertEqual(exp_v, x._values())

        # Duplicate indices
        i = self.IndexTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.ValueTensor([3, 2, 4, 1])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.IndexTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.ValueTensor([6, 4])

        x = self.safeCoalesce(x)
        self.assertEqual(exp_i, x._indices())
        self.assertEqual(exp_v, x._values())

    def test_contig_hybrid(self):
        i = self.IndexTensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = self.ValueTensor([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        ])
        x = self.SparseTensor(i, v, torch.Size([100, 100, 2]))
        exp_i = self.IndexTensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = self.ValueTensor([
            [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
            [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
        ])
        x = self.safeCoalesce(x)
        self.assertEqual(exp_i, x._indices())
        self.assertEqual(exp_v, x._values())

        i = self.IndexTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.ValueTensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.ValueTensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]])

        x = self.safeCoalesce(x)
        self.assertEqual(exp_i, x._indices())
        self.assertEqual(exp_v, x._values())

        # Duplicate indices
        i = self.IndexTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.ValueTensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.IndexTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.ValueTensor([[6, 4, 5], [4, 3, 4]])

        x = self.safeCoalesce(x)
        self.assertEqual(exp_i, x._indices())
        self.assertEqual(exp_v, x._values())

    def test_clone(self):
        x, _, _ = self._gen_sparse(4, 20, 5)
        if self.is_uncoalesced:
            self.assertFalse(x.is_coalesced())
            y = x.clone()
            self.assertFalse(y.is_coalesced())
        x = x.coalesce()
        self.assertTrue(x.is_coalesced())
        y = x.clone()
        self.assertTrue(y.is_coalesced())

    def test_transpose(self):
        x = self._gen_sparse(4, 20, 5)[0]
        y = x.to_dense()

        for i, j in itertools.combinations(range(4), 2):
            x = x.transpose_(i, j)
            y = y.transpose(i, j)
            self.assertEqual(x.to_dense(), y)

            x = x.transpose(i, j)
            y = y.transpose(i, j)
            self.assertEqual(x.to_dense(), y)

    @cpu_only
    def test_mm(self):
        def test_shape(di, dj, dk):
            x, _, _ = self._gen_sparse(2, 20, [di, dj])
            t = torch.randn(di, dk)
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.addmm(alpha, t, beta, x, y)
            expected = torch.addmm(alpha, t, beta, x.to_dense(), y)
            self.assertEqual(res, expected)

            res = torch.addmm(t, x, y)
            expected = torch.addmm(t, x.to_dense(), y)
            self.assertEqual(res, expected)

            res = torch.mm(x, y)
            expected = torch.mm(x.to_dense(), y)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100)
        test_shape(100, 1000, 200)
        test_shape(64, 10000, 300)

    @cpu_only
    def test_saddmm(self):
        def test_shape(di, dj, dk):
            x = self._gen_sparse(2, 20, [di, dj])[0]
            t = self._gen_sparse(2, 20, [di, dk])[0]
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.saddmm(alpha, t, beta, x, y)
            expected = torch.addmm(alpha, t.to_dense(), beta, x.to_dense(), y)
            self.assertEqual(res.to_dense(), expected)

            res = torch.saddmm(t, x, y)
            expected = torch.addmm(t.to_dense(), x.to_dense(), y)
            self.assertEqual(res.to_dense(), expected)

            res = torch.smm(x, y)
            expected = torch.mm(x.to_dense(), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3)
        test_shape(1000, 100, 100)
        test_shape(3000, 64, 300)

    def test_dsmm(self):
        def test_shape(di, dj, dk):
            x = self._gen_sparse(2, 20, [di, dj])[0]
            y = self.randn(dj, dk)

            res = torch.dsmm(x, y)
            expected = torch.mm(x.to_dense(), y)
            self.assertEqual(res, expected)

        test_shape(7, 5, 3)
        test_shape(1000, 100, 100)
        test_shape(3000, 64, 300)

    def test_hsmm(self):
        def test_shape(di, dj, dk):
            x = self._gen_sparse(2, 20, [di, dj])[0]
            y = self.randn(dj, dk)

            res = torch.hsmm(x, y)
            expected = torch.mm(x.to_dense(), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3)
        test_shape(1000, 100, 100)
        test_shape(3000, 64, 300)

    def _test_spadd_shape(self, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x, _, _ = self._gen_sparse(len(shape_i), 10, shape)
        y = self.randn(*shape)
        r = random.random()

        res = torch.add(y, r, x)
        expected = y + r * x.to_dense()

        self.assertEqual(res, expected)

        # Non contiguous dense tensor
        s = list(shape)
        s[0] = shape[-1]
        s[-1] = shape[0]
        y = self.randn(*s)
        y.transpose_(0, len(s) - 1)
        r = random.random()

        res = torch.add(y, r, x)
        expected = y + r * x.to_dense()

        self.assertEqual(res, expected)

    def test_spadd(self):
        self._test_spadd_shape([5, 6])
        self._test_spadd_shape([10, 10, 10])
        self._test_spadd_shape([50, 30, 20])
        self._test_spadd_shape([5, 5, 5, 5, 5, 5])

    def test_spadd_hybrid(self):
        self._test_spadd_shape([5, 6], [2, 3])
        self._test_spadd_shape([10, 10, 10], [3])
        self._test_spadd_shape([50, 30, 20], [2])
        self._test_spadd_shape([5, 5, 5, 5, 5, 5], [2])

    def _test_basic_ops_shape(self, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), 9, shape)
        x2, _, _ = self._gen_sparse(len(shape_i), 12, shape)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = x1.to_dense() + x2.to_dense()
        self.assertEqual(y1.to_dense(), expected)
        self.assertEqual(y2.to_dense(), expected)

        y1 = x1 - x2
        y2 = x1.clone()
        y2.sub_(x2)
        expected = x1.to_dense() - x2.to_dense()
        self.assertEqual(y1.to_dense(), expected)
        self.assertEqual(y2.to_dense(), expected)

        y1 = x1 * x2
        y2 = x1.clone()
        y2.mul_(x2)
        expected = x1.to_dense() * x2.to_dense()
        self.assertEqual(y1.to_dense(), expected)
        self.assertEqual(y2.to_dense(), expected)

        y1 = x1 * 37.5
        y2 = x1.clone()
        y2.mul_(37.5)
        expected = x1.to_dense() * 37.5
        self.assertEqual(y1.to_dense(), expected)
        self.assertEqual(y2.to_dense(), expected)

        y1 = x1 / 37.5
        y2 = x1.clone()
        y2.div_(37.5)
        expected = x1.to_dense() / 37.5
        self.assertEqual(y1.to_dense(), expected)
        self.assertEqual(y2.to_dense(), expected)

        # TODO: add back inplace support
        y1 = x1 ** 2
        y2 = x1.clone()
        y2 = y2.pow(2)
        expected = x1.to_dense() ** 2
        self.assertEqual(y1.to_dense(), expected)
        self.assertEqual(y2.to_dense(), expected)

        y = x1.clone()
        y.zero_()
        expected = torch.zeros(x1.size())
        self.assertEqual(y.to_dense(), expected)

        self.assertFalse(x1.is_coalesced())
        y = x1.coalesce()
        z = x1.coalesce()
        self.assertFalse(x1.is_coalesced())
        self.assertTrue(y.is_coalesced())
        self.assertEqual(x1, y)
        # check that coalesce is out of place
        y._values().add_(1)
        self.assertEqual(z._values() + 1, y._values())

    def test_basic_ops(self):
        self._test_basic_ops_shape([5, 6])
        self._test_basic_ops_shape([10, 10, 10])
        self._test_basic_ops_shape([50, 30, 20])
        self._test_basic_ops_shape([5, 5, 5, 5, 5, 5])

    def test_basic_ops_hybrid(self):
        self._test_basic_ops_shape([5, 6], [2, 3])
        self._test_basic_ops_shape([10, 10, 10], [3])
        self._test_basic_ops_shape([50, 30, 20], [2])
        self._test_basic_ops_shape([5, 5, 5, 5, 5, 5], [2])

    def _test_sparse_mask_shape(self, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x, _, _ = self._gen_sparse(len(shape_i), 9, shape)
        x = x.coalesce()
        dense = self.randn(*shape)
        dense._sparse_copy_(x)
        y = dense._sparse_mask(x)
        self.assertEqual(x, y)

    def _test_sparse_mask_fixed(self):
        i = self.IndexTensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.ValueTensor([100, 200, 300, 400])
        x = self.SparseTensor(i, v, torch.Size([5, 4])).coalesce()
        dense = self.ValueTensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ])
        exp_v = self.ValueTensor([7, 14, 3, 20])
        res = dense._sparse_mask(x)
        expected = self.SparseTensor(i, exp_v, torch.Size([5, 4]))
        self.assertEqual(res, expected)

        expected_dense = self.ValueTensor([
            [1, 2, 300, 4],
            [5, 6, 100, 8],
            [9, 10, 11, 12],
            [13, 200, 15, 16],
            [17, 18, 19, 400],
        ])
        dense._sparse_copy_(x)
        self.assertEqual(dense, expected_dense)

    def prop_sparse_mask_const(self, x, s):
        """sparse_mask does not mutate inputs."""
        expected_x = x.clone()
        expected_s = s.clone()
        try:
            x._sparse_mask(s)
        except:
            pass
        self.assertEqual(x, expected_x)
        self.assertEqual(s, expected_s)

    def _sparse_mask_py(self, x, s):
        """Python reference implementation of sparse_mask"""
        if not s.is_coalesced():
            raise ValueError("mask is not coalesced")
        if x.size() != s.size():
            raise ValueError("sparse_mask operands have incompatible sizes")
        ss = SparseSize.from_sparse(s).setSize(x.size())
        r = ss.new_sparse(dense2sparse(type(x)), s._indices())
        v = r._values()
        for i, js in enumerate(s._indices().t()):
            if any(j < 0 for j in js):
                raise IndexError("negative index is out of range")
            # js is a tensor, and indexing by a tensor does something different
            v[i] = x[tuple(js)]
        return r

    def prop_sparse_mask_py(self, x, s):
        """sparse_mask agrees with reference implementation"""
        # NB: It would be better if we could make more precise statements
        # about what exceptions should be raised if there is an error,
        # but the different APIs are actually quite inconsistent here.
        expect_error = False
        error = False
        expected_r = None
        try:
            expected_r = self._sparse_mask_py(x.clone(), s.clone())
        except (ValueError, RuntimeError, IndexError) as e:
            expect_error = e
        try:
            r = x._sparse_mask(s)
        except (ValueError, RuntimeError, IndexError) as e:
            error = e
        if expect_error and not error:
            self.fail("Expected an error {:s}, but no error was thrown".format(repr(expect_error)))
        elif not expect_error and error:
            self.fail("Did not expect error, but {:s} was thrown".format(repr(error)))
        elif not expect_error and not error:
            self.assertEqual(r, expected_r)

    def prop_sparse_mask(self, x, s):
        self.prop_sparse_mask_const(x.clone(), s.clone())
        self.prop_sparse_mask_py(x.clone(), s.clone())

    def _mk_sparse(self, i, v, size=None):
        if isinstance(size, list):
            size = torch.Size(size)
        s = self.SparseTensor(self.IndexTensor(i), self.ValueTensor(v), size)
        return s

    def _test_sparse_mask_oob(self):
        def go(d, i, v, s, well_formed=False):
            # CUDA has stronger invariants on sparse tensors being well-formed,
            # so we can't easily test for OOB without triggering asserts
            # earlier.  If we tighten up the CPU code, we probably have to get
            # rid of these tests entirely.
            if not well_formed and self.is_cuda:
                return
            sparse = self._mk_sparse(i, v, s)
            if well_formed:
                sparse.to_dense() # check that dims make sense
            self.prop_sparse_mask(d, sparse.coalesce())

        dense = self.ValueTensor([88,99])
        go(dense, [[0,1]],  [88,99], [2], True)

        dense = self.ValueTensor([42])
        go(dense, [[0]],    [42], [1], True)
        go(dense, [[9999]], [0],  [1])
        go(dense, [[1]],    [0],  [1])
        go(dense, [[-1]],   [0],  [1])
        go(dense, [[1]],    [0],  [2], True)

        dense = self.ValueTensor([
            [1, 2, 3],
            [4, 5, 6]
        ])
        go(dense, [[0]],    [[1, 2, 3]], [2, 3], True)
        go(dense, [[1]],    [[4, 5, 6]], [2, 3], True)
        go(dense, [[2]],    [[0, 0, 0]], [2, 3])
        go(dense, [[-1]],   [[0, 0, 0]], [2, 3])
        go(dense, [[0],
                   [0]],    [1], [2, 3], True)
        go(dense, [[1],
                   [2]],    [6], [2, 3], True)
        go(dense, [[0],
                   [3]],    [0], [2, 3])

    def test_sparse_mask(self):
        self._test_sparse_mask_fixed()
        self._test_sparse_mask_oob()

        self._test_sparse_mask_shape([5, 6])
        self._test_sparse_mask_shape([10, 10, 10])
        self._test_sparse_mask_shape([50, 30, 20])
        self._test_sparse_mask_shape([5, 5, 5, 5, 5, 5])

    def _test_sparse_mask_hybrid_fixed(self):
        i = self.IndexTensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.ValueTensor([[100, 200], [200, 300], [300, 400], [400, 500]])
        # TODO: This is also testing that, if coalesce is a no-op,
        # the indices don't get permuted. I don't know if we actually
        # want to give this invariant.
        x = self.SparseTensor(i, v, torch.Size([5, 4, 2])).coalesce()
        dense = self.ValueTensor([
            [[1, 3], [2, 2], [3, 3], [4, 2]],
            [[5, 7], [6, 7], [7, 9], [8, 9]],
            [[9, 2], [10, 4], [11, 1], [12, 3]],
            [[13, 5], [14, 1], [15, 1], [16, 6]],
            [[17, 7], [18, 2], [19, 7], [20, 1]],
        ])
        res = dense._sparse_mask(x)
        exp_v = self.ValueTensor([[7, 9], [14, 1], [3, 3], [20, 1]])
        expected = self.SparseTensor(i, exp_v, torch.Size([5, 4, 2]))
        self.assertEqual(res, expected)

        expected_dense = self.ValueTensor([
            [[1, 3], [2, 2], [300, 400], [4, 2]],
            [[5, 7], [6, 7], [100, 200], [8, 9]],
            [[9, 2], [10, 4], [11, 1], [12, 3]],
            [[13, 5], [200, 300], [15, 1], [16, 6]],
            [[17, 7], [18, 2], [19, 7], [400, 500]],
        ])
        dense._sparse_copy_(x)
        self.assertEqual(dense, expected_dense)


    def test_sparse_mask_hybrid(self):
        self._test_sparse_mask_hybrid_fixed()

        self._test_sparse_mask_shape([5, 6], [2, 3])
        self._test_sparse_mask_shape([10, 10, 10], [3])
        self._test_sparse_mask_shape([50, 30, 20], [2])
        self._test_sparse_mask_shape([5, 5, 5, 5, 5, 5], [2])


class TestUncoalescedSparse(TestSparse):
    def setUp(self):
        super(TestUncoalescedSparse, self).setUp()
        self.is_uncoalesced = True


@unittest.skipIf(not TEST_CUDA, 'CUDA not available')
class TestCudaSparse(TestSparse):
    def setUp(self):
        super(TestCudaSparse, self).setUp()
        self.is_cuda = True
        self.IndexTensor = torch.cuda.LongTensor
        self.ValueTensor = torch.cuda.DoubleTensor
        self.SparseTensor = torch.cuda.sparse.DoubleTensor


@unittest.skipIf(not TEST_CUDA, 'CUDA not available')
class TestCudaUncoalescedSparse(TestCudaSparse):
    def setUp(self):
        super(TestCudaUncoalescedSparse, self).setUp()
        self.is_uncoalesced = True

if __name__ == '__main__':
    run_tests()
