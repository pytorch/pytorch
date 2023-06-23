# Owner(s): ["module: sparse"]

import torch
import itertools
import functools
import operator
import random
import unittest
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfRocm, do_test_dtypes, \
    load_tests, TEST_NUMPY, TEST_SCIPY, IS_WINDOWS, gradcheck, coalescedonoff, \
    DeterministicGuard, first_sample, TEST_WITH_CROSSREF, TEST_WITH_ROCM, skipIfTorchDynamo, \
    parametrize, subtest, is_coalesced_indices, suppress_warnings, instantiate_parametrized_tests, \
    skipIfCrossRef
from torch.testing._internal.common_cuda import TEST_CUDA
from numbers import Number
from typing import Dict, Any
from distutils.version import LooseVersion
from torch.testing._internal.common_cuda import \
    (SM53OrLater, SM80OrLater)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, dtypesIfCUDA, onlyCPU, onlyCUDA, precisionOverride,
     deviceCountAtLeast, OpDTypes, onlyNativeDeviceTypes)
from torch.testing._internal.common_methods_invocations import \
    (op_db, reduction_ops, sparse_unary_ufuncs, sparse_masked_reduction_ops, binary_ufuncs)
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex, all_types_and_complex_and, floating_and_complex_types,
    floating_and_complex_types_and, integral_types, floating_types_and,
)
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse


def _op_supports_any_sparse(op):
    return (op.supports_sparse
            or op.supports_sparse_csr
            or op.supports_sparse_csc
            or op.supports_sparse_bsr
            or op.supports_sparse_bsc)


reduction_ops_with_sparse_support = [op for op in reduction_ops if 'masked.' not in op.name and _op_supports_any_sparse(op)]

binary_ufuncs_with_sparse_support = [op for op in binary_ufuncs if _op_supports_any_sparse(op)]

like_fns_with_sparse_support = [op for op in op_db if _op_supports_any_sparse(op) and '_like' in op.name]

if TEST_SCIPY:
    import scipy.sparse

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# batched grad doesn't support sparse
gradcheck = functools.partial(gradcheck, check_batched_grad=False)

CUSPARSE_SPMM_COMPLEX128_SUPPORTED = (
    IS_WINDOWS and torch.version.cuda and LooseVersion(torch.version.cuda) > "11.2"
) or (not IS_WINDOWS and not TEST_WITH_ROCM)

def all_sparse_layouts(test_name='layout', include_strided=False):
    return parametrize(test_name, [
        subtest(torch.strided, name='Strided'),
        subtest(torch.sparse_coo, name='SparseCOO'),
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC'),
        subtest(torch.sparse_bsr, name='SparseBSR'),
        subtest(torch.sparse_bsc, name='SparseBSC'),
    ][(0 if include_strided else 1):])

def gradcheck_semantics(test_name='gradcheck'):
    gradcheck_sparse = functools.partial(gradcheck, masked=False)
    gradcheck_masked = functools.partial(gradcheck, masked=True)
    gradcheck_sparse.masked = False
    gradcheck_masked.masked = True
    return parametrize(test_name, [
        subtest(gradcheck_sparse, name='sparse'),
        subtest(gradcheck_masked, name='masked')])


class CrossRefSparseFakeMode(torch._subclasses.CrossRefFakeMode):
    def __init__(self):
        super().__init__(
            self.ignore_op, check_strides=False,
            check_aliasing=False,
        )  # TODO: enable stride/alias checking

    # empty_like excluded for now due to sparse complex
    # aten._to_dense.default this one is getting called with csc
    @staticmethod
    def ignore_op(func):
        return func in (
            torch.ops.aten.empty_like.default,
            torch.ops.aten.set_.source_Storage_storage_offset,
            torch.ops.aten.sspaddmm.out,
            torch.ops.aten._spdiags.default,
            torch.ops.aten._to_dense.default,
            torch.ops.aten.indices.default,
            torch.ops.aten._indices.default,
            torch.ops.aten.values.default,
            torch.ops.aten._values.default,
        )

class TestSparseLegacyAndDeprecation(TestCase):

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_legacy_warnings(self):

        def f1():
            "torch.sparse.SparseTensor() is deprecated."\
                "  Please use torch.sparse_coo_tensor((0,), dtype=)"
            x_ref = torch.sparse_coo_tensor((0,), dtype=torch.float64)
            x = torch.sparse.DoubleTensor()
            self.assertEqual(x, x_ref)

        def f2():
            "torch.sparse.SparseTensor(cdata=x._cdata) is deprecated."\
                "  Please use torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)"
            x_ref = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64).to_sparse()
            x = torch.sparse.DoubleTensor(cdata=x_ref._cdata)
            y = torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)
            self.assertEqual(x, x_ref)
            self.assertEqual(y, x_ref)

        def f3():
            "torch.sparse.SparseTensor(indices, values, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], dtype=torch.float64)
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64))
            self.assertEqual(x, x_ref)

        def f4():
            "torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], (2, 3), dtype=torch.float64)
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64), (2, 3))
            self.assertEqual(x, x_ref)

        def f5():
            "torch.sparse.SparseTensor(shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(shape, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor((2, 3), dtype=torch.float64)
            x = torch.sparse.DoubleTensor(2, 3)
            self.assertEqual(x, x_ref)

        for test_f in [f1, f2, f3, f4, f5]:

            with self.assertWarns(UserWarning, msg=test_f.__doc__) as cm:
                test_f()
                test_f()

            # Check warn-once:
            self.assertEqual(len(cm.warnings), 1)

    @parametrize('fast_mode', (True, False))
    def test_gradcheck_check_sparse_nnz(self, fast_mode):
        """Tests for deprecated check_sparse_nnz keyword argument of gradcheck.

        Deprecation steps:
        2.1: Specification of check_sparse_nnz triggers a warning.
        2.2: Specification of check_sparse_nnz triggers an
             exception. Remove all check_sparse_nnz usages from
             gradcheck and delete this test.
        """
        def fn(x, masked_grad):
            return x.to_dense(masked_grad=masked_grad)

        def test(x, masked_grad, masked, check_sparse_nnz):
            x = x.detach().clone().requires_grad_()
            torch.autograd.gradcheck(fn, (x, masked_grad), masked=masked, check_sparse_nnz=check_sparse_nnz, fast_mode=fast_mode)

        x = torch.tensor([[0, 2], [3, 4]], dtype=torch.float64).to_sparse()

        for masked_grad, masked, check_sparse_nnz in itertools.product(*[(True, False, None)] * 3):
            effective_masked_grad = True if masked_grad is None else masked_grad
            effective_check_sparse_nnz = False if check_sparse_nnz is None else check_sparse_nnz
            # For BC, the effective masked depends on the value of specified check_sparse_nnz:
            effective_masked = (check_sparse_nnz if check_sparse_nnz is not None else False) if masked is None else masked

            warn_using_check_sparse_nnz = self.assertWarns(
                UserWarning,
                msg=('Backwards compatibility: check_sparse_nnz is deprecated, it will be removed in a future version of PyTorch.'
                     f' Use masked={effective_check_sparse_nnz} instead.'))
            raise_on_non_equal_masked_and_check_sparse_nnz = self.assertRaisesRegex(
                ValueError,
                f"Expected specified check_sparse_nnz [(]={effective_check_sparse_nnz}[)]"
                f" to be equal to masked [(]={effective_masked}[)]")
            raise_jacobian_mismatch = self.assertRaisesRegex(RuntimeError, "Jacobian mismatch for output 0 with respect to input 0")

            def run_test():
                if effective_masked_grad != effective_masked and not fast_mode:
                    with raise_jacobian_mismatch:
                        test(x, masked_grad, masked, check_sparse_nnz)
                else:
                    test(x, masked_grad, masked, check_sparse_nnz)

            if masked != check_sparse_nnz and None not in {masked, check_sparse_nnz}:
                # the specified masked and check_sparse_nnz must match
                with warn_using_check_sparse_nnz:
                    with raise_on_non_equal_masked_and_check_sparse_nnz:
                        test(x, masked_grad, masked, check_sparse_nnz)
            elif check_sparse_nnz is not None:
                with warn_using_check_sparse_nnz:
                    run_test()
            else:
                self.assertNotWarn(run_test)

class TestSparseBase(TestCase):
    def run(self, result=None):
        if TEST_WITH_CROSSREF:
            with CrossRefSparseFakeMode():
                return super().run(result)
        else:
            return super().run(result)

class TestSparse(TestSparseBase):

    def setUp(self):
        TestCase.setUp(self)

        self.index_tensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs, dtype=torch.int64)

        def sparse_empty_factory(*args, **kwargs):
            kwargs['layout'] = kwargs.get('layout', torch.sparse_coo)
            return torch.empty(*args, **kwargs)
        self.sparse_empty = sparse_empty_factory

        def sparse_tensor_factory(*args, **kwargs):
            return torch.sparse_coo_tensor(*args, **kwargs)
        self.sparse_tensor = sparse_tensor_factory

    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.genSparseTensor(with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device)

        if not coalesced:
            self.assert_uncoalesced(x)

        return x, i, v

    def assert_uncoalesced(self, x):
        """
        Test if a CPU tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        assert not x.is_coalesced()
        existing_indices = set()
        for i in range(x._nnz()):
            index = str(x._indices()[:, i])
            if index in existing_indices:
                return True
            else:
                existing_indices.add(index)

    def randn(self, *args, **kwargs):
        """
        Variant of torch.randn that also works in the TEST_CUDA case.
        """
        # TODO: Put this in torch.cuda.randn
        return torch.empty(*args, **kwargs).normal_()

    @dtypes(torch.double)
    def test_print_coalesced(self, device, dtype):
        self._test_print(device, dtype, True)

    @dtypes(torch.double)
    def test_print_uncoalesced(self, device, dtype):
        self._test_print(device, dtype, False)

    def _test_print(self, device, dtype, coalesced):
        shape_sparse_dim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]
        printed = []
        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            indices_shape = torch.Size((sparse_dim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            printed.append("# shape: {}".format(torch.Size(shape)))
            printed.append("# nnz: {}".format(nnz))
            printed.append("# sparse_dim: {}".format(sparse_dim))
            printed.append("# indices shape: {}".format(indices_shape))
            printed.append("# values shape: {}".format(values_shape))

            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=device).view(indices_shape)
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if not coalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=dtype,
                                  device=device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.sparse_tensor(indices, values, shape, dtype=dtype, device=device)

            dtypes = [torch.int32]
            if values.dtype == torch.double:
                dtypes.append(torch.float)
            else:
                dtypes.append(torch.double)
            for dtype in dtypes:
                printed.append("########## {} ##########".format(dtype))
                x = sp_tensor.detach().to(dtype)
                printed.append("# sparse tensor")
                printed.append(str(x))
                if x.dtype.is_floating_point:
                    printed.append("# after requires_grad_")
                    printed.append(str(x.requires_grad_()))
                    printed.append("# after addition")
                    printed.append(str(x + x))
                printed.append("# _indices")
                printed.append(str(x._indices()))
                printed.append("# _values")
                printed.append(str(x._values()))
            printed.append('')
        self.assertExpected('\n'.join(printed))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_basic(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            self.assertEqual(i, x._indices())
            self.assertEqual(v, x._values())
            self.assertEqual(x.ndimension(), len(with_size))
            self.assertEqual(x.coalesce()._nnz(), nnz if x.is_coalesced() else nnz // 2)
            self.assertEqual(list(x.size()), with_size)

            # Test .indices() and .values()
            if not coalesced:
                with self.assertRaisesRegex(RuntimeError, "Cannot get indices on an uncoalesced tensor"):
                    x.indices()
                with self.assertRaisesRegex(RuntimeError, "Cannot get values on an uncoalesced tensor"):
                    x.values()
            else:
                self.assertEqual(x.indices(), x._indices())
                self.assertEqual(x.values(), x._values())

        test_shape(3, 10, 100)
        test_shape(3, 10, [100, 100, 100])
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

        # Make sure that coalesce handles duplicate indices correctly
        i = self.index_tensor([[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]], device=device)
        v = torch.tensor([[idx**2, idx] for idx in range(i.size(1))], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([10, 2]), dtype=dtype, device=device)
        self.assertEqual(x.coalesce()._nnz(), 9)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 1e-2})
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    def test_coalesce(self, device, dtype, coalesced):

        def _test_coalesce(t):
            tc = t.coalesce()
            self.assertEqual(tc.to_dense(), t.to_dense())
            self.assertTrue(tc.is_coalesced())
            # Our code below doesn't work when nnz is 0, because
            # then it's a 0D tensor, not a 2D tensor.
            if t._nnz() == 0:
                self.assertEqual(t._indices(), tc._indices())
                self.assertEqual(t._values(), tc._values())
                return tc

            value_map: Dict[Any, Any] = {}
            for idx, val in zip(t._indices().t(), t._values()):
                idx_tup = tuple(idx.tolist())
                if idx_tup in value_map:
                    value_map[idx_tup] += val
                else:
                    value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

            new_indices = sorted(value_map.keys())
            _new_values = [value_map[idx] for idx in new_indices]
            if t._values().ndimension() < 2:
                new_values = t._values().new(_new_values)
            else:
                new_values = torch.stack(_new_values)

            new_indices = t._indices().new(new_indices).t()
            tg = t.new(new_indices, new_values, t.size())

            self.assertEqual(tc._indices(), tg._indices())
            self.assertEqual(tc._values(), tg._values())

            if t.is_coalesced():
                self.assertEqual(tc._indices(), t._indices())
                self.assertEqual(tc._values(), t._values())

        for empty_i, empty_v, empty_nnz in itertools.product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5

            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size + dense_size, dtype, device, coalesced)
            _test_coalesce(t)  # this tests correctness

    @dtypes(torch.double)
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/89395")
    def test_coalesce_reference_cycle(self, device, dtype):
        # Test coalesce doesn't create autograd graph cycles (gh-52253)

        # Sanity check that the helper class works as expected
        t = torch.rand(2)
        t_ref = torch._C._WeakTensorRef(t)
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        def test_sparse_sum():
            i = torch.tensor([[0], [4]], dtype=torch.long, device=device)
            v = torch.tensor([[[-0.4567, -1.8797, 0.0380, 1.4316]]],
                             dtype=dtype, device=device)
            S = torch.sparse_coo_tensor(i, v)
            S = S.coalesce()
            S.requires_grad_(True)
            S2 = S.coalesce()
            self.assertTrue(S2.is_coalesced())
            return torch._C._WeakTensorRef(S2)

        ref = test_sparse_sum()
        self.assertTrue(ref.expired())

    @dtypes(torch.double)
    def test_ctor_large_sizes(self, device, dtype):
        # Test that integer overflow is detected when computing numel
        # of a sparse tensor with large dimensions (gh-57416). Notice
        # that numel is computed internally when constructing a
        # tensor, hence the overflow may appear during the tensor
        # construction step.
        N = 100000
        indices = torch.tensor([[N, N - 1]] * 4, dtype=torch.int64, device=device)
        values = torch.tensor([1, 2], dtype=dtype, device=device)
        self.assertRaises(RuntimeError,
                          lambda: torch.sparse_coo_tensor(
                              indices, values, (N + 1,) * 4, device=device))

    @dtypes(torch.double, torch.cdouble)
    def test_ctor_size_checks(self, device, dtype):
        indices = self.index_tensor([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], device=device)
        values = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)

        # indices inconsistent with size
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 1, 1])))

        # values inconsistent with size
        values = torch.tensor([
            [2, 1, 2, 1],
            [1, 0, 5, 2],
        ], dtype=dtype, device=device)
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 4, 2, 1])))

    @dtypes(*floating_and_complex_types_and(torch.float16, torch.bfloat16))
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    @gradcheck_semantics()
    def test_to_dense_with_gradcheck(self, device, dtype, gradcheck):

        def test_tensor(x, res):
            x.to_dense()  # Tests triple to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            dense_x = x.to_dense()
            safe_dense_x = self.safeToDense(x)
            dense_x = dense_x.to(res.dtype)
            safe_dense_x = safe_dense_x.to(res.dtype)
            self.assertEqual(res, dense_x)
            self.assertEqual(res, safe_dense_x)

            # Only run autograd test for float64
            if x.dtype != torch.float64:
                return

            def fn(x):
                return x.to_dense(masked_grad=gradcheck.masked)
            x.requires_grad_(True)
            gradcheck(fn, (x,))

        for value_type in [torch.double, torch.cdouble]:
            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            # we don't have to_dense for half types on CPU because it is implemented
            # with a slower add_ operation
            v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]), dtype=value_type, device=device)
            res = torch.tensor([
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
            ], dtype=dtype, device=device)

            test_tensor(x, res)
            test_tensor(res, res)

            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            v = torch.empty(4, 0, dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]), dtype=value_type, device=device)
            res = torch.empty((3, 4, 5, 0), dtype=dtype, device=device)
            test_tensor(x, res)

    @coalescedonoff
    @dtypes(torch.float16, torch.bfloat16, torch.float64, torch.int, torch.cfloat, torch.cdouble)
    def test_to_sparse(self, device, dtype, coalesced):
        shape = [5, 2, 10, 4]
        max_nnz = 1
        for value_type in [torch.double, torch.cdouble]:
            for dim, dim_sz in enumerate(shape, 1):
                max_nnz *= dim_sz
                rnnz = torch.randint(2, max_nnz, (1,)).item()
                for nnz in [0, 1, rnnz]:
                    expected, _, _ = self._gen_sparse(dim, nnz, shape, dtype=value_type, device=device,
                                                      coalesced=coalesced)
                    expected = expected.to(dtype)

                    d = expected.to_dense()
                    result = d.to_sparse(dim)
                    self.assertEqual(d, result.to_dense())
                    self.assertEqual(expected.size(), result.size())
                    self.assertEqual(dim, result.sparse_dim())

    @dtypes(torch.double, torch.cdouble)
    def test_sparse_bool(self, device, dtype):
        a = torch.tensor([True, False], dtype=dtype, device=device).to(torch.bool)
        b = a.to_sparse().to_dense()
        self.assertEqual(a, b)

    @dtypes(torch.double, torch.cdouble)
    def test_scalar(self, device, dtype):
        # tensor with value
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1), 12.3, [], dtype=dtype, device=device)
        self.assertEqual(1, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

        # tensor with multiple values
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1).expand(0, 2),
                               [12.3, 12.3], [], dtype=dtype, device=device)
        self.assertEqual(2, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3 * 2, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a.coalesce(), a.coalesce().to_dense().to_sparse())

        # tensor without value
        a = self.sparse_empty((), dtype=dtype, device=device)
        self.assertEqual(0, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(0, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

    @dtypes(torch.double, torch.cdouble)
    def test_shared(self, device, dtype):
        i = self.index_tensor([[2]], device=device)
        v = torch.tensor([5], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3]))
        v[0] = 6
        self.assertEqual(torch.tensor([0, 0, 6], dtype=dtype, device=device), self.safeToDense(x))
        i[0][0] = 0
        self.assertEqual(torch.tensor([6, 0, 0], dtype=dtype, device=device), self.safeToDense(x))

        i = self.index_tensor([[2]], device=device)
        v = torch.empty((1, 0), dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        i[0][0] = 0
        self.assertEqual(torch.empty((3, 0), dtype=dtype, device=device), self.safeToDense(x))

    @dtypes(torch.double, torch.cdouble)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    @gradcheck_semantics()
    def test_to_dense_hybrid(self, device, dtype, gradcheck):

        def test_tensor(x, res):
            x.to_dense()  # Tests double to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())
            self.assertEqual(res, self.safeToDense(x))

            def fn(x):
                return x.to_dense(masked_grad=gradcheck.masked)
            x.requires_grad_(True)
            gradcheck(fn, (x,))

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        v = torch.tensor([[2, 3], [1, 2], [3, 4], [4, 5]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2]))
        res = torch.tensor([
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
        ], dtype=dtype, device=device)
        test_tensor(x, res)

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        v = torch.empty((4, 2, 0), dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2, 0]))
        res = torch.empty((3, 4, 2, 0), dtype=dtype, device=device)
        test_tensor(x, res)

    @dtypes(torch.double, torch.cdouble)
    def test_contig(self, device, dtype):
        def test_tensor(x, exp_i, exp_v):
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], device=device)
        v = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([100, 100]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], device=device)
        exp_v = torch.tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.empty([4, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.tensor([6, 4], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.empty([2, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

    @dtypes(torch.double, torch.cdouble)
    def test_contig_hybrid(self, device, dtype):
        def test_tensor(x, exp_i, exp_v):
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], device=device)
        v = torch.tensor([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        ], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([100, 100, 2]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], device=device)
        exp_v = torch.tensor([
            [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
            [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
        ], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.tensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.tensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.tensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.tensor([[6, 4, 5], [4, 3, 4]], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.empty([2, 3, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_clone(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            if not coalesced:
                self.assertFalse(x.is_coalesced())
                y = x.clone()
                self.assertFalse(y.is_coalesced())
            x = x.coalesce()
            self.assertTrue(x.is_coalesced())
            y = x.clone()
            self.assertTrue(y.is_coalesced())

        test_shape(4, 20, 5)
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 2e-2})
    def test_Sparse_to_Sparse_copy_(self, device, dtype, coalesced):
        # This is for testing torch.copy_(SparseTensor, SparseTensor)
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)

        # test copy
        x2_dense = x2.to_dense()
        x1.copy_(x2)
        self.assertEqual(x2_dense, x1.to_dense())

        # test type conversion (when x1.copy_(x2), x1.dtype should stay the same)
        x1 = x1.to(torch.float32)

        x2 = x2.to(torch.float16)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        x2 = x2.to(torch.float64)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        # test no broadcast
        self.assertRaises(RuntimeError, lambda: x1.copy_(x2.narrow_copy(0, 0, 1)))

        # test raise error on copy_() between dense and sparse Tensors
        self.assertRaises(RuntimeError, lambda: x1.copy_(torch.randn(5, 5)))

        # test autograd
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone()
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to_dense())
        self.assertEqual(None, x1.grad)

    @coalescedonoff
    @unittest.skipIf(torch.cuda.device_count() < 2, "no multi-GPU")
    @dtypes(torch.double, torch.cdouble)
    def test_Sparse_to_Sparse_copy_multi_gpu(self, device, dtype, coalesced):
        # This is for testing torch.copy_(SparseTensor, SparseTensor) across GPU devices
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x1 = x1.to('cuda:0')

        def test_cross_device(x1, x2):
            x1_device = x1.device
            x1.copy_(x2)
            self.assertEqual(x2.to('cuda:0').to_dense(), x1.to_dense())
            self.assertEqual(x1_device, x1.device)

        test_cross_device(x1, x2.to('cuda:1'))  # test across gpu devices
        test_cross_device(x1, x2.to('cpu'))  # test between cpu and gpu

        # test autograd
        x2 = x2.to('cuda:1')
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone().to('cuda:0')
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to('cuda:0').to_dense())
        self.assertEqual(None, x1.grad)

    @onlyCUDA
    def test_cuda_empty(self, device):
        def test_tensor(x):
            y = x.to(device)
            self.assertEqual(x.sparse_dim(), y.sparse_dim())
            self.assertEqual(x.dense_dim(), y.dense_dim())
            x = y.cpu()
            self.assertEqual(y.sparse_dim(), x.sparse_dim())
            self.assertEqual(y.dense_dim(), x.dense_dim())

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float32)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4, 0), dtype=torch.float32)
        test_tensor(x)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_transpose(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            y = self.safeToDense(x)

            for i, j in itertools.combinations(range(4), 2):
                x = x.transpose_(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

                x = x.transpose(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

        test_shape(4, 6, 3)
        test_shape(4, 3, [7, 7, 7, 3, 3, 3, 0])
        test_shape(4, 0, [0, 0, 7, 3, 3, 3, 0])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    @gradcheck_semantics()
    def test_permute(self, device, dtype, coalesced, gradcheck):
        # trivial checks
        s = torch.rand(3, 3, 3, device=device, dtype=dtype).to_sparse()
        with self.assertRaisesRegex(RuntimeError, "does not match the length"):
            s.permute(dims=(1, 0))
        with self.assertRaisesRegex(RuntimeError, "duplicate dims"):
            s.permute(dims=(1, 1, 1))

        def test_shape(sparse_dims, nnz, with_size):
            ndim = len(with_size)
            valid_sparse_dims = torch.arange(-ndim, -ndim + sparse_dims)
            valid_dense_dims = torch.arange(-ndim + sparse_dims, 0)

            for dims in itertools.permutations(range(-ndim, 0)):
                s = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
                d = self.safeToDense(s)

                dims_sparse, _ = torch.tensor(dims[:sparse_dims]).sort()
                dims_dense, _ = torch.tensor(dims[sparse_dims:]).sort()

                if (valid_sparse_dims == dims_sparse).all() and (valid_dense_dims == dims_dense).all():
                    # if valid permutation, test for correctness
                    s_permuted = s.permute(dims)
                    self.assertEqual(s_permuted, d.permute(dims))

                    # if s is coalesced, and perm does not touch 0-dim,
                    # the result has to be coalesced as well
                    if dims[0] == 0:
                        self.assertEqual(s_permuted.is_coalesced(), s.is_coalesced())
                    else:
                        self.assertFalse(s_permuted.is_coalesced())

                    gradcheck(lambda t: t.permute(dims).to_dense(masked_grad=gradcheck.masked), s.requires_grad_())
                else:
                    # otherwise check if exception is thrown
                    fail_message = "transpositions between sparse and dense dimensions are not allowed"
                    with self.assertRaisesRegex(RuntimeError, fail_message):
                        s.permute(dims)

        test_shape(2, 3, [2, 3, 4, 5])
        test_shape(2, 3, [2, 2, 0])
        # if nnz=0, it is not true that t == t.to_dense().to_sparse()
        # unless t.sparse_dim == t.dim (i.e. t is not hybrid)
        test_shape(3, 0, [0, 0, 2])

    @coalescedonoff
    @onlyCPU
    @dtypes(torch.double)
    def test_coalesce_transpose_mm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [dj, di], dtype, device, coalesced)
            y = torch.randn(dj, dk, dtype=dtype, device=device)

            x_coalesced = x.coalesce()
            self.assertTrue(x_coalesced.is_coalesced())

            x_coalesced_t = x_coalesced.t()
            # Transpose is `colasced`-preserving if the indices tensor is empty.
            self.assertEqual(x_coalesced_t.is_coalesced(), di * nnz == 0)

            res = torch.mm(x_coalesced_t, y)
            expected = torch.mm(self.safeToDense(x_coalesced_t), y)
            self.assertEqual(res, expected)

        test_shape(10, 20, 30, 20)
        test_shape(0, 20, 30, 0)
        test_shape(10, 0, 30, 0)
        test_shape(10, 20, 0, 0)
        test_shape(10, 20, 0, 20)

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1166")
    @dtypes(torch.double, torch.cdouble)
    def test_t_empty(self, device, dtype):
        def test_in_place(x):
            shape_original = x.shape
            x.t_()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), x.size())
            self.assertEqual(0, x._indices().numel())
            self.assertEqual(0, x._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        def test_not_in_place(x):
            shape_original = x.shape
            y = x.t()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), y.size())
            self.assertEqual(0, y._indices().numel())
            self.assertEqual(0, y._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        x = self.sparse_empty(2, 3, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

        x = self.sparse_empty(2, 0, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_add_zeros(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, sizes):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            zeros = torch.sparse_coo_tensor(sizes, device=x.device)
            r1 = zeros + x
            r2 = x + zeros
            self.assertEqual(r1, x)
            self.assertEqual(r2, x)

        test_shape(1, 20, [1])
        test_shape(4, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 0])

    @dtypes(torch.double, torch.cdouble)
    def test_add_sub_nnz(self, device, dtype):
        # nnz should not grow unbounded (gh-34964)
        x = torch.randn(10, dtype=dtype, device=device).to_sparse()
        x.add_(x)
        x.add_(x)
        self.assertLessEqual(x._nnz(), 10)

        x.sub_(2 * x)
        x.sub_(2 * x)
        self.assertLessEqual(x._nnz(), 10)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_cat(self, device, dtype, coalesced):
        # shapes: list of tuples (sparse_dims, nnz, sizes)
        def test_shapes(shapes, dim, fail_message=None):
            inputs = [self._gen_sparse(shape[0], shape[1], shape[2], dtype, device, coalesced)[0]
                      for shape in shapes]
            if fail_message:
                with self.assertRaisesRegex(RuntimeError, fail_message):
                    torch.cat(inputs, dim)
            else:
                result = torch.cat(inputs, dim)
                dense_result = torch.cat([t.to_dense() for t in inputs], dim)
                self.assertEqual(dense_result, result.to_dense())

        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], 1)

        # mismatched sizes
        test_shapes([(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4])], 0,
                    "All tensors must have the same shape: \\[2, 3, 4].*\\[2, 1, 4]")
        # hybrid sparse/dense
        test_shapes(
            [(2, 10, [2, 3, 4]), (2, 10, [2, 1, 4]), (2, 10, [2, 4, 4])], 1)
        # cat along dense dim
        test_shapes([(2, 10, [2, 3, 4]), (2, 10, [2, 3, 7])], 2)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 1)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 2)
        # mismatched dimensions
        test_shapes([(2, 10, [2, 3, 4]), (3, 10, [2, 3, 4])], 0,
                    "All tensors must have the same.*2, 1, but tensor at position 1 has 3, 0.")
        # wrapped dimension
        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], -2)

        # sparse with dense
        sp = self._gen_sparse(3, 10, [2, 3, 4], dtype, device, coalesced)[0]
        dn = sp.to_dense()
        with self.assertRaisesRegex(RuntimeError,
                                    "Concatenating sparse tensors, but a dense tensor was found at position 1."):
            torch.cat((sp, dn))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_unsqueeze(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, sizes, unsqueeze_dim, fail_message=None):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.unsqueeze(x, unsqueeze_dim)
            else:
                result = torch.unsqueeze(x, unsqueeze_dim)
                dense_result = torch.unsqueeze(x.to_dense(), unsqueeze_dim)
                self.assertEqual(dense_result, result.to_dense())

        # basic case
        test_shape(3, 10, [5, 7, 11], 0)

        # hybrid sparse/dense, unsqueeze along sparse dim
        test_shape(3, 10, [5, 7, 11, 13, 17], 0)
        test_shape(3, 10, [5, 7, 11, 13, 17], 3)

        # unsqueeze along dense dimensions
        test_shape(3, 10, [5, 7, 11, 13, 17], 4)
        test_shape(3, 10, [5, 7, 11, 13, 17], 5)

        # wrapped dimensions
        test_shape(3, 10, [5, 7, 11, 13, 17], -1)
        test_shape(3, 10, [5, 7, 11, 13, 17], -6)

        # bounds
        test_shape(3, 10, [5, 7, 11, 13, 17], -7, "Dimension out of range")
        test_shape(3, 10, [5, 7, 11, 13, 17], 6, "Dimension out of range")

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_select(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.select(x, select_dim, select_index)
            else:
                result = torch.select(x, select_dim, select_index)
                if result.is_sparse:
                    result = result.to_dense()
                dense_result = torch.select(x.to_dense(), select_dim, select_index)
                self.assertEqual(dense_result, result)


        sizes = [5, 7, 11, 13, 17]
        # hybrid sparse/dense, select sparse dim, result is dense
        for i in range(sizes[0]):
            test_shape(1, 10, sizes, 0, i)
        test_shape(1, 10, sizes, 0, sizes[0] + 1, r'select[(][)][:] index \d out of range.*')

        # hybrid sparse/dense, select sparse dim, result is sparse
        for d in range(3):
            for i in range(sizes[d]):
                test_shape(3, 10, sizes, d, i)

        # hybrid sparse/dense, select dense dim, result is sparse
        for d in range(1, 3):
            for i in range(sizes[d]):
                test_shape(1, 10, sizes, d, i)

    @dtypes(*integral_types())
    def test_select_no_type_promotion(self, device, dtype):
        # see https://github.com/pytorch/pytorch/issues/82150
        idx = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
        val = torch.ones(6, dtype=dtype)
        s = torch.sparse_coo_tensor(idx, val, size=(3, 3))

        for t in (s, s * torch.tensor(0, dtype=dtype)):
            # empty checks
            self.assertEqual(t.dtype, t[2].dtype)
            self.assertEqual(t.dtype, t[0, 1].dtype)
            # sum should not promote
            self.assertEqual(t.dtype, t[0, 0].dtype)
            self.assertEqual(t.dtype, t[1, 1].dtype)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            if isinstance(select_index, int):
                select_index = [select_index]
            if isinstance(select_index, list):
                select_index = torch.tensor(select_index, device=device, dtype=torch.long)
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.index_select(x, select_dim, select_index)
            else:
                result = torch.index_select(x, select_dim, select_index)
                if result.is_sparse:
                    result = result.to_dense()
                dense_result = torch.index_select(x.to_dense(), select_dim, select_index)
                self.assertEqual(dense_result, result)

        sizes = [5, 7, 11, 13, 17]
        for d in range(len(sizes)):
            for index in [0, sizes[d] - 1, [0, sizes[d] // 2, sizes[d] - 1]]:
                test_shape(1, 10, sizes, d, index)
                test_shape(len(sizes) // 2, 10, sizes, d, index)
                test_shape(len(sizes), 10, sizes, d, index)

    def _test_index_select_exhaustive_index(self, sizes, dims, device, dtype, coalesced):
        t = make_tensor(sizes, dtype=dtype, device=device)
        t_sparse = t.to_sparse().coalesce() if coalesced else t.to_sparse()
        t_small_sparse, _, _ = self._gen_sparse(len(sizes), 2, sizes, dtype, device, coalesced)
        t_small = t_small_sparse.to_dense()
        for d in dims:
            # NOTE: indices are negative
            idx_dim_d_range = list(range(-sizes[d], 0))
            for idx_len in range(sizes[d], sizes[d] + 1):
                # creates all possible valid indices into dim d of lenght idx_len
                for idx in itertools.product(*itertools.repeat(idx_dim_d_range, idx_len)):
                    t_idx = torch.tensor(idx, dtype=torch.long, device=device)

                    # NOTE: index_select for dense does not support negative indices,
                    # hence + sizes[d]. See https://github.com/pytorch/pytorch/issues/76347

                    # tests the nnz > sizes[d] branch
                    dense_result = t.index_select(d, t_idx + sizes[d])
                    sparse_result = t_sparse.index_select(d, t_idx)
                    self.assertEqual(dense_result, sparse_result)

                    # tests the nnz <= sizes[d] branch
                    small_dense_result = t_small.index_select(d, t_idx + sizes[d])
                    small_sparse_result = t_small_sparse.index_select(d, t_idx)
                    self.assertEqual(small_dense_result, small_sparse_result)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select_exhaustive_index_small(self, device, dtype, coalesced):
        # will trigger brute-force algo
        self._test_index_select_exhaustive_index((3, 3, 4), range(3), device, dtype, coalesced)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select_exhaustive_index_large(self, device, dtype, coalesced):
        # will trigger more sophisticated algos
        self._test_index_select_exhaustive_index((100, 50, 3, 3), (2, 3), device, dtype, coalesced)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select_empty_and_non_contiguous_index(self, device, dtype, coalesced):
        # empty index
        idx_empty = torch.tensor([], dtype=torch.long, device=device)
        t = make_tensor((5, 5), dtype=dtype, device=device)
        res_dense = t.index_select(0, idx_empty)
        res_sparse = t.to_sparse().index_select(0, idx_empty)
        self.assertEqual(res_dense, res_sparse)

        # non-contigous index
        idx = torch.randint(low=0, high=5, size=(10, 2), device=device)[:, 0]

        def run_test(sizes):
            # case nnz > size[d]
            t = make_tensor(sizes, dtype=dtype, device=device)
            res_dense = t.index_select(0, idx)
            res_sparse = t.to_sparse().index_select(0, idx)
            self.assertEqual(res_dense, res_sparse)

            # case nnz <= size[d]
            t_small_sparse, _, _ = self._gen_sparse(len(sizes), 2, sizes, dtype, device, coalesced)
            res_sparse = t_small_sparse.index_select(0, idx)
            res_dense = t_small_sparse.to_dense().index_select(0, idx)
            self.assertEqual(res_dense, res_sparse)

        # brute-force
        run_test((10, 10))
        # more sophisticated algos
        run_test((10, 100, 100))

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select_parallelization(self, device, dtype, coalesced):
        """
        Test with sizes that will trigger parallelization (i.e. with sizes
        that are >= at::internal::GRAIN_SIZE)
        """
        def run_test(nnz, size):
            t_sparse, _, _ = self._gen_sparse(1, nnz, (size,), dtype, device, coalesced)
            t_dense = t_sparse.to_dense()

            # idx_small to (sort) and (binary) search into t_sparse
            idx_small = torch.randint(size, (nnz // 2,), device=device)
            # idx_large to (sort) and (binary) search into idx_large
            # NOTE: when coalesced=True, the (binary) search will be
            # done over t_sparse anyway, as it is already sorted.
            idx_large = torch.randint(size, (nnz * 2,), device=device)
            for idx in (idx_small, idx_large):
                res_dense = t_dense.index_select(0, idx)
                res_sparse = t_sparse.index_select(0, idx)
                self.assertEqual(res_dense, res_sparse)

        # NOTE: GRAIN_SIZE = 32768
        # case nnz <= size[d]
        tlen = 70000  # > 2 * GRAIN_SIZE
        run_test(tlen, tlen)

        # case nnz > size[d]
        run_test(tlen, tlen // 2)

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_mm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)
            t = torch.randn(di, dk, dtype=dtype, device=device)
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            alpha = random.random()
            beta = random.random()

            res = torch.addmm(t, x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(t, self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(res, expected)

            res = torch.addmm(t, x, y)
            expected = torch.addmm(t, self.safeToDense(x), y)
            self.assertEqual(res, expected)

            res = torch.mm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 200, 20)
        test_shape(64, 10000, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 100, 0)
        test_shape(10, 100, 0, 0)
        test_shape(10, 100, 0, 20)

    @unittest.skipIf(
        IS_WINDOWS and TEST_CUDA,
        "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1"
    )
    @coalescedonoff
    @dtypes(torch.double)
    def test_bmm(self, device, dtype, coalesced):
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            a_list = []
            b_list = []
            for mat_idx in range(num_mats):
                a_mat = self._gen_sparse(2, nnz, [dim_i, dim_j], dtype, device, coalesced)[0]
                b_mat = torch.randn([dim_j, dim_k], dtype=dtype, device=device)
                a_list.append(a_mat)
                b_list.append(b_mat)

            a = torch.stack(a_list)
            b = torch.stack(b_list)
            ab = a.bmm(b)

            # Compare each matrix against result from mm()
            for mat_idx in range(num_mats):
                a_mat = a_list[mat_idx]
                b_mat = b_list[mat_idx]
                ab_mat_bmm = ab[mat_idx]
                ab_mat_mm = a_mat.mm(b_mat)
                self.assertEqual(ab_mat_bmm, ab_mat_mm)

        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

        a = torch.rand([10, 23, 32], dtype=dtype, device=device)
        a[3] = torch.zeros(23, 32, dtype=dtype, device=device)
        a[6] = torch.zeros(23, 32, dtype=dtype, device=device)
        a = a.to_sparse()
        b = torch.rand([10, 32, 10], dtype=dtype, device=device)
        b[4] = torch.zeros(32, 10, dtype=dtype, device=device)
        b[6] = torch.zeros(32, 10, dtype=dtype, device=device)
        ab = a.bmm(b)
        for mat_idx in range(ab.size(0)):
            ab_mat = ab[mat_idx]
            ab_mat_check = a[mat_idx].mm(b[mat_idx])
            self.assertEqual(ab_mat, ab_mat_check)

        ab_traspose_check = b.transpose(1, 2).to_sparse().bmm(
            a.transpose(1, 2).to_dense()
        ).transpose(1, 2)
        self.assertEqual(ab, ab_traspose_check)

    @onlyCUDA
    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(
        IS_WINDOWS,
        "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1"
    )
    def test_bmm_deterministic(self, device, dtype, coalesced):
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            a_list = []
            b_list = []
            for mat_idx in range(num_mats):
                a_list.append(self._gen_sparse(2, nnz, [dim_i, dim_j], dtype, device, coalesced)[0])
                b_list.append(torch.randn([dim_j, dim_k], dtype=dtype, device=device))

            a = torch.stack(a_list).cuda()
            b = torch.stack(b_list).cuda()
            with DeterministicGuard(torch.are_deterministic_algorithms_enabled()):
                torch.use_deterministic_algorithms(False)
                ab_nondeterministic = torch.bmm(a, b)
                torch.use_deterministic_algorithms(True)
                ab_deterministic = torch.bmm(a, b)
            diff_abs = (ab_deterministic - ab_nondeterministic).abs()
            diff_rel = diff_abs / ab_deterministic.abs()
            diff_rel[torch.isnan(diff_rel)] = 0

            # deterministic and non-deterministic results should either be
            # equal or within a small relative difference
            equal_abs_or_rel = diff_abs.eq(0).logical_or(diff_rel.lt(0.001))
            self.assertTrue(equal_abs_or_rel.all())

        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

    @onlyCUDA
    @unittest.skipIf(
        not IS_WINDOWS or not TEST_WITH_ROCM,
        "this test ensures bmm sparse-dense CUDA gives an error when run on Windows with CUDA < 11.0"
    )
    @dtypes(torch.double)
    def test_bmm_windows_error(self, device, dtype):
        a = torch.rand(2, 2, 2, dtype=dtype).to_sparse().cuda()
        b = torch.rand(2, 2, 2, dtype=dtype).cuda()
        with self.assertRaisesRegex(
                RuntimeError,
                "bmm sparse-dense CUDA is not supported on Windows with cuda before 11.0"):
            ab = a.bmm(b)

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_saddmm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            t = self._gen_sparse(2, nnz, [di, dk], dtype, device, coalesced)[0]
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            alpha = random.random()
            beta = random.random()

            res = torch.saddmm(t, x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.saddmm(t, x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.smm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

    @onlyCPU
    @coalescedonoff
    # adding a graph break before self.assertFalse(weight._indices().is_contiguous())
    # makes the test pass so some existent sparse related bug
    @skipIfTorchDynamo("skip")
    @dtypes(torch.double, torch.cdouble)
    def test_sspaddmm(self, device, dtype, coalesced):

        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            t = self._gen_sparse(2, nnz, [di, dk], dtype, device, coalesced)[0]
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            alpha = random.random()
            beta = random.random()

            res = t.sspaddmm(x, y, beta=beta, alpha=alpha)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            self.assertEqual(self.safeToDense(res), expected)

            res = t.sspaddmm(x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

        # Test code from issue https://github.com/pytorch/pytorch/issues/45113
        batch_size, input_size, hidden_size = 5, 3, 7

        # Create coalesced sparse tensor with non-contiguous indices
        weight = torch.randn(hidden_size, input_size, dtype=dtype, device=device).to_sparse()
        self.assertTrue(weight.is_coalesced())
        non_contig_indices = weight.indices().mT.contiguous().mT
        weight = torch.sparse_coo_tensor(
            indices=non_contig_indices, values=weight.values(), size=weight.shape)
        weight._coalesced_(True)
        self.assertFalse(weight._indices().is_contiguous())
        # Create un/coalesced sparse tensor
        bias = torch.randn((hidden_size, 1), dtype=dtype, device=device).to_sparse()
        bias = torch.cat([bias] * batch_size, dim=1)

        if coalesced:
            bias = bias.coalesce()

        x = torch.randn(input_size, batch_size, dtype=dtype, device=device)
        res = bias.sspaddmm(weight, x)

        true_result = (bias.to_dense() + torch.matmul(weight.to_dense(), x)).to_sparse()
        self.assertEqual(self.safeToDense(res), self.safeToDense(true_result))

    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    def test_sparse_addmm(self, device, dtype, coalesced):
        if dtype is torch.bfloat16:
            # RuntimeError: "addmm_sparse_dense" not implemented for 'BFloat16'
            self.skipTest('See https://github.com/pytorch/pytorch/issues/73145')

        def test_shape(m, n, p, nnz, broadcast, alpha_beta=None):
            if alpha_beta is None:
                alpha = random.random()
                beta = random.random()
            else:
                alpha, beta = alpha_beta
            if broadcast:
                D1 = make_tensor((), dtype=dtype, device=device, requires_grad=True)
            else:
                D1 = make_tensor([n, p], dtype=dtype, device=device, requires_grad=True)
            D2 = make_tensor([m, p], dtype=dtype, device=device, requires_grad=True)
            S = self._gen_sparse(2, nnz, [n, m], dtype, device, coalesced)[0]
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            Y = torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            Y_dense = torch.addmm(D1, S_dense, D2, beta=beta, alpha=alpha)
            self.assertEqual(Y, Y_dense)

            def fn(S, D1, D2, beta=beta, alpha=alpha):
                return torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            gradcheck(fn, (S, D1, D2), masked=True)

        test_shape(7, 8, 9, 20, False, None)
        test_shape(7, 8, 9, 20, True, None)
        test_shape(7, 8, 9, 20, False, (1, 0))
        test_shape(7, 8, 9, 20, True, (1, 0))
        test_shape(7, 8, 9, 20, False, (1, 1))
        test_shape(7, 8, 9, 20, True, (1, 1))

    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    def test_sparse_mm(self, device, dtype, coalesced):
        def test_shape(d1, d2, d3, nnz, transposed):
            if transposed:
                D = torch.randn(d3, d2, dtype=dtype,
                                device=device).t_().requires_grad_(True)
            else:
                D = torch.randn(d2, d3, dtype=dtype, device=device).requires_grad_(True)
            S = self._gen_sparse(2, nnz, [d1, d2], dtype, device, coalesced)[0]
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

            def fn(S, D):
                return torch.sparse.mm(S, D)
            gradcheck(fn, (S, D), masked=True)

        test_shape(7, 8, 9, 20, False)
        test_shape(7, 8, 9, 20, True)

    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    @gradcheck_semantics()
    def test_sparse_mul(self, device, dtype, coalesced, gradcheck):
        # https://github.com/pytorch/pytorch/issues/79914
        a = torch.tensor([[0., 1]], dtype=dtype, device=device).to_sparse().requires_grad_(True)
        b = torch.tensor([[0., 1]], dtype=dtype, device=device).to_sparse().requires_grad_(True)
        gradcheck(lambda x, y: torch.sparse.sum(x * y).to_dense(masked_grad=gradcheck.masked), [a, b])

        def test_shape(sparse_dims, nnz, with_shape):
            a = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[0].requires_grad_(True)
            b = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[0].requires_grad_(True)

            self.assertEqual((a * b).to_dense(), a.to_dense() * b.to_dense(), masked=True)
            gradcheck(lambda x, y: (x * y).to_dense(), [a, b])
            # Issues with 0-dim indices/values
            gradcheck(lambda x, y: torch.sparse.sum(x * y).to_dense(), [a, b], masked=True)

        # TODO: Re-enable these
        # test_shape(2, 3, [2, 3, 4, 5])
        # test_shape(2, 3, [2, 2, 0])

    @coalescedonoff
    @dtypes(torch.double)
    def test_dsmm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            y = self.randn(dj, dk, dtype=dtype, device=device)

            res = torch.dsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    @coalescedonoff
    @dtypes(torch.double)
    def test_hsmm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            y = self.randn(dj, dk, dtype=dtype, device=device)

            res = torch.hsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    @coalescedonoff
    @dtypes(torch.double)
    def test_spadd(self, device, dtype, coalesced):

        def _test_spadd_shape(nnz, shape_i, shape_v=None):
            shape = shape_i + (shape_v or [])
            x, _, _ = self._gen_sparse(len(shape_i), nnz, shape, dtype, device, coalesced)
            y = self.randn(*shape, dtype=dtype, device=device)
            r = random.random()

            res = torch.add(y, x, alpha=r)
            expected = y + r * self.safeToDense(x)

            self.assertEqual(res, expected)

            # Non contiguous dense tensor
            s = list(shape)
            s[0] = shape[-1]
            s[-1] = shape[0]
            y = self.randn(*s, dtype=dtype, device=device)
            y.transpose_(0, len(s) - 1)
            r = random.random()

            res = torch.add(y, x, alpha=r)
            expected = y + r * self.safeToDense(x)

            self.assertEqual(res, expected)

            x, i, v = self._gen_sparse(len(shape_i), nnz, shape, dtype, device, coalesced)
            nnz = i.size(1)

            # Non contiguous sparse indices tensor
            x_ = self.sparse_tensor(i[:, ::2], v[:(nnz + 1) // 2], x.shape, dtype=dtype, device=device)
            res = torch.add(y, x_, alpha=r)
            expected = y + r * self.safeToDense(x_)
            self.assertEqual(res, expected)

            # Non contiguous sparse values tensor

            x_ = self.sparse_tensor(i[:, :(nnz + 1) // 2], v[::2], x.shape, dtype=dtype, device=device)
            res = torch.add(y, x_, alpha=r)
            expected = y + r * self.safeToDense(x_)
            self.assertEqual(res, expected)

            # Non contiguous sparse indices and values tensors
            x_ = self.sparse_tensor(i[:, 1::2], v[1::2], x.shape, dtype=dtype, device=device)
            res = torch.add(y, x_, alpha=r)
            expected = y + r * self.safeToDense(x_)
            self.assertEqual(res, expected)

        def _test_spadd():
            _test_spadd_shape(10, [5, 6])
            _test_spadd_shape(10, [10, 10, 10])
            _test_spadd_shape(10, [50, 30, 20])
            _test_spadd_shape(10, [5, 5, 5, 5, 5, 5])
            _test_spadd_shape(0, [0, 30, 20])
            _test_spadd_shape(0, [50, 0, 20])
            _test_spadd_shape(0, [50, 30, 0])

        def _test_spadd_hybrid():
            _test_spadd_shape(10, [5, 6], [2, 3])
            _test_spadd_shape(10, [10, 10, 10], [3])
            _test_spadd_shape(10, [50, 30, 20], [2])
            _test_spadd_shape(10, [5, 5, 5, 5, 5, 5], [2])
            _test_spadd_shape(0, [0, 30, 20], [2, 0])
            _test_spadd_shape(0, [50, 0, 20], [2, 0])
            _test_spadd_shape(0, [50, 30, 0], [2, 0])
            _test_spadd_shape(10, [50, 30, 20], [2, 0])

        _test_spadd()
        _test_spadd_hybrid()

    @coalescedonoff
    @dtypes(torch.float)
    def test_sparse_add_out_bfloat16(self, device, dtype, coalesced):
        # fp32
        x, _, _ = self._gen_sparse(3, 5, 10, dtype, device, coalesced)
        y, _, _ = self._gen_sparse(3, 5, 10, dtype, device, coalesced)
        res_fp32 = torch.add(x, y)

        # bfloat16
        x = x.bfloat16()
        y = y.bfloat16()
        res_bf16 = torch.add(x, y)
        res_bf16 = res_bf16.float()  # to compare with reference
        self.assertEqual(res_fp32, res_bf16, atol=1e-2, rtol=0)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_norm(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            y = x.coalesce()
            self.assertEqual(x.norm(), y._values().norm())

        test_shape(3, 10, 100)
        test_shape(4, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(4, 0, [0, 0, 100, 5, 5, 5, 0])

        # Unsupported arguments should error
        kwarg_error_pairs = [
            ({'keepdim': True},
             RuntimeError, r'norm_sparse currently does not support keepdim=True'),
            ({'dim': 0},
             RuntimeError, r'norm_sparse currently only supports full reductions'),
            ({'dtype': torch.double, 'p': 'fro'},
             ValueError, r'dtype argument is not supported in frobenius norm'),
            ({'dtype': torch.double, 'p': 0},
             RuntimeError, r"norm_sparse currently does not support 'dtype' argument")
        ]
        x = self._gen_sparse(3, 10, 100, dtype, device, coalesced)[0]
        for kwargs, err, msg in kwarg_error_pairs:
            with self.assertRaisesRegex(err, msg):
                x.norm(**kwargs)

    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(TEST_WITH_CROSSREF, "fallback triggers cuda device error")
    def test_sparse_sum(self, device, dtype, coalesced):

        def run_tests(S, td=None):
            D = S.coalesce().to_dense().detach().requires_grad_(True)
            if td is None:
                S_sum = torch.sparse.sum(S)
                D_sum = D.sum()
                self.assertEqual(S_sum.item(), D_sum.item())

                def fn(S):
                    return torch.sparse.sum(S)
                gradcheck(fn, (S,), masked=True)
            else:
                S_sum = torch.sparse.sum(S, td)
                D_sum = D.sum(td)
                self.assertEqual(S_sum.to_dense() if S_sum.is_sparse else S_sum, D_sum)

                def fn(S):
                    res = torch.sparse.sum(S, td)
                    return res.to_dense(masked_grad=True)
                gradcheck(fn, (S,), masked=True)

        nnz = 10
        sparse_dims = 2
        with_size = [5, 5, 1, 4]  # use a dense dim = 1 to test for squeeze
        test_dims = []
        for i in range(1, 5):
            test_dims += itertools.combinations(range(len(with_size)), i)

        # https://github.com/pytorch/pytorch/issues/16501
        x = torch.tensor([[1., 0., 0., 1.],
                          [0., 1., 0., 0.],
                          [0., 1., 1., 0.],
                          [0., 1., 0., 2.]], dtype=dtype, device=device).to_sparse()
        self.assertEqual(torch.sparse.sum(x, dim=0), torch.sparse.sum(x, dim=-2))
        self.assertEqual(torch.sum(x.to_dense(), dim=0), torch.sparse.sum(x, dim=0).to_dense())

        S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]

        # dim out of range
        self.assertRaises(IndexError, lambda: torch.sparse.sum(S, 5))

        # dim 0 appears multiple times in the list of dims
        self.assertRaises(RuntimeError, lambda: torch.sparse.sum(S, [0, 0]))

        # sum an empty tensor
        empty_S = torch.sparse_coo_tensor(size=with_size, dtype=dtype, device=device)
        self.assertEqual(torch.sparse.sum(empty_S, [0]).to_dense(), torch.sum(empty_S.to_dense(), [0]))
        self.assertEqual(torch.sparse.sum(empty_S), torch.tensor(0, dtype=dtype, device=device))
        empty_S.requires_grad_(True)
        empty_S_sum = torch.sparse.sum(empty_S)
        empty_S_sum.backward()
        self.assertEqual(empty_S.grad.to_dense(), empty_S.clone().detach().to_dense())

        # test values().sum()
        S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
        run_tests(S.requires_grad_(True))

        for test_dim in test_dims:
            S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            run_tests(S.requires_grad_(True), test_dim)

    def _test_basic_ops_shape(self, nnz_x1, nnz_x2, shape_i, shape_v, dtype, device, coalesced):
        shape = shape_i + (shape_v)
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape, dtype, device, coalesced)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 - x2
        y2 = x1.clone()
        y2.sub_(x2)
        expected = self.safeToDense(x1) - self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * x2
        y2 = x1.clone()
        y2.mul_(x2)
        expected = self.safeToDense(x1) * self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * 37.5
        y2 = x1.clone()
        y2.mul_(37.5)
        expected = self.safeToDense(x1) * 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 / 37.5
        y2 = x1.clone()
        y2.div_(37.5)
        expected = self.safeToDense(x1) / 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 // 37.5
        y2 = x1.clone()
        y2.floor_divide_(37.5)
        expected = self.safeToDense(x1) // 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # TODO: add back inplace support
        y1 = x1 ** 2
        y2 = x1.clone()
        y2 = y2.pow(2)
        expected = self.safeToDense(x1) ** 2
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y = x1.clone()
        y.zero_()
        expected = torch.zeros(x1.size(), dtype=dtype, device=device)
        self.assertEqual(self.safeToDense(y), expected)

        self.assertEqual(x1.is_coalesced(), coalesced)
        y = x1.coalesce()
        z = x1.coalesce()
        self.assertEqual(x1.is_coalesced(), coalesced)
        self.assertTrue(y.is_coalesced())
        y._values().add_(1)
        if not x1.is_coalesced():
            # check that coalesce is out of place if the original tensor is not
            # coalesced.
            self.assertEqual(z._values() + 1, y._values())
        else:
            # check that coalesce is in-place if the original tensor is
            # coalesced.
            self.assertEqual(z._values(), y._values())

    @coalescedonoff
    @dtypes(torch.double)
    def test_basic_ops(self, device, dtype, coalesced):

        def _test_basic_ops():
            self._test_basic_ops_shape(9, 12, [5, 6], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [50, 30, 20], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 0], [], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [], [], dtype, device, coalesced)

        def _test_basic_ops_hybrid():
            self._test_basic_ops_shape(9, 12, [5, 6], [2, 3], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [3], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [50, 30, 20], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [2], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
            self._test_basic_ops_shape(0, 0, [10, 10, 0], [2, 0], dtype, device, coalesced)

        _test_basic_ops()
        _test_basic_ops_hybrid()

    @dtypes(torch.double, torch.cdouble)
    def test_add_dense_sparse_mismatch(self, device, dtype):
        def test_shape(dense_size, sparse_dims_shape, dense_dims_shape, sparse_size):
            x = torch.zeros(dense_size, dtype=dtype, device=device)
            sparse_y = self.sparse_tensor(torch.zeros(sparse_dims_shape, dtype=torch.int64, device=device),
                                          torch.randn(dense_dims_shape, dtype=dtype, device=device),
                                          torch.Size(sparse_size))
            with self.assertRaisesRegex(
                    RuntimeError,
                    "add: expected 'self' and 'other' to have same size"):
                x + sparse_y

        test_shape([3, 4], [1, 4], [4, 4, 4], [3, 4, 4])
        test_shape([3, 4, 0], [1, 4], [4, 4, 4, 0], [3, 4, 4, 0])

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @dtypes(torch.double, torch.cdouble)
    def test_add_noncontiguous(self, device, dtype):
        indices = self.index_tensor([[1, 2], [0, 2]], device=device)
        values = torch.tensor([1.], dtype=dtype, device=device).expand(2, 3, 4, 5)
        x = self.sparse_tensor(indices, values, dtype=dtype, device=device)
        assert not x._values().is_contiguous()
        y = x + x
        expected = self.safeToDense(x) + self.safeToDense(x)
        self.assertEqual(self.safeToDense(y), expected)

    def _test_sparse_mask_shape(self, nnz_x1, nnz_x2, shape_i, shape_v, dtype, device, coalesced):
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape, dtype, device, coalesced)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

    @dtypes(torch.double, torch.cdouble)
    @skipIfCrossRef
    def test_sparse_mask_backward(self, device, dtype):
        from itertools import product, repeat

        shape = (5, 5)
        sparse_dims = len(shape)
        nnzs = (0, 5, 15, 25)

        lhs_data = torch.arange(1, 26, device=device).reshape(shape).to(dtype).to_sparse(sparse_dims)
        rhs_data = lhs_data.clone()

        for nnz in nnzs:
            for lhs_is_coalesced, rhs_is_coalesced in product(*repeat((True, False), 2)):
                lhs = torch.sparse_coo_tensor(
                    lhs_data._indices()[:, :nnz],
                    lhs_data._values()[:nnz],
                    lhs_data.shape
                )._coalesced_(lhs_is_coalesced).requires_grad_(True)

                rhs = torch.sparse_coo_tensor(
                    lhs_data._indices()[:, -nnz:],
                    lhs_data._values()[-nnz:],
                    lhs_data.shape
                )._coalesced_(rhs_is_coalesced).requires_grad_(True)

                # setting masked = True is required because of the broken backward of to_dense().
                # See https://github.com/pytorch/pytorch/issues/95550.
                gradcheck(lambda x, y: x.sparse_mask(y).to_dense(), (lhs, rhs), masked=True, check_sparse_nnz=True)
                gradcheck(lambda x, y: x.sparse_mask(y).to_dense(), (lhs, lhs.detach()), masked=True, check_sparse_nnz=True)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_mask(self, device, dtype, coalesced):
        def _test_sparse_mask_fixed():
            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ], device=device)
            v = torch.tensor([1, 2, 3, 4], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([5, 4]), dtype=dtype, device=device).coalesce()
            dense = torch.tensor([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ], dtype=dtype, device=device)
            exp_v = torch.tensor([7, 14, 3, 20], dtype=dtype, device=device)
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse()
            res_sparse_lhs = sparse.sparse_mask(x)
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4]), dtype=dtype, device=device)
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag.
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ], device=device)
            v = torch.empty([4, 0], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([5, 4, 0])).coalesce()
            dense = torch.empty([5, 4, 0], dtype=dtype, device=device)
            exp_v = torch.empty([4, 0], dtype=dtype, device=device)
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse(2)
            res_sparse_lhs = sparse.sparse_mask(x)
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 0]), dtype=dtype, device=device)
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag.
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

        _test_sparse_mask_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [50, 30, 20], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 0], [], dtype, device, coalesced)

        # check repetitions and matchings in the intersection
        lhs = torch.randint(0, 5, (100,), device=device)
        rhs = torch.randint(0, 5, (100,), device=device).to_sparse()
        self.assertEqual(lhs.to_sparse().sparse_mask(rhs), lhs.sparse_mask(rhs))

        # check coalesce
        sparse_c = torch.rand(3, 3, device=device).to_sparse()
        sparse_unc = torch.rand(3, 3, device=device).to_sparse()._coalesced_(False)
        for lhs, rhs in [(sparse_c, sparse_unc), (sparse_unc, sparse_c)]:
            res_all_sparse = lhs.sparse_mask(rhs)
            res_dense_sparse = lhs.to_dense().sparse_mask(rhs)
            self.assertEqual(res_all_sparse.coalesce(), res_dense_sparse.coalesce())
            self.assertEqual(rhs.is_coalesced(), res_all_sparse.is_coalesced())

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_mask_hybrid(self, device, dtype, coalesced):
        def _test_sparse_mask_hybrid_fixed():
            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ])
            v = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]])
            # TODO: This is also testing that, if coalesce is a no-op,
            # the indices don't get permuted. I don't know if we actually
            # want to give this invariant.
            x = self.sparse_tensor(i, v, torch.Size([5, 4, 2])).coalesce()
            dense = torch.tensor([
                [[1, 3], [2, 2], [3, 3], [4, 2]],
                [[5, 7], [6, 7], [7, 9], [8, 9]],
                [[9, 2], [10, 4], [11, 1], [12, 3]],
                [[13, 5], [14, 1], [15, 1], [16, 6]],
                [[17, 7], [18, 2], [19, 7], [20, 1]],
            ])
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse(2)
            res_sparse_lhs = sparse.sparse_mask(x)
            exp_v = torch.tensor([[7, 9], [14, 1], [3, 3], [20, 1]])
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 2]))
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

            i = self.index_tensor([
                [1, 3, 0, 4],
                [2, 1, 2, 3],
            ])
            v = torch.empty(4, 2, 0)
            x = self.sparse_tensor(i, v, torch.Size([5, 4, 2, 0])).coalesce()
            dense = torch.empty(5, 4, 2, 0)
            res_dense_lhs = dense.sparse_mask(x)
            sparse = dense.to_sparse(2)
            res_sparse_lhs = sparse.sparse_mask(x)
            exp_v = torch.empty(4, 2, 0)
            expected = self.sparse_tensor(i, exp_v, torch.Size([5, 4, 2, 0]))
            self.assertEqual(res_dense_lhs.coalesce(), expected.coalesce())
            # check no side effects for the coalesce flag
            self.assertTrue(sparse.is_coalesced())
            self.assertEqual(res_sparse_lhs.coalesce(), expected.coalesce())

        _test_sparse_mask_hybrid_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6], [2, 3], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [3], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [50, 30, 20], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
        self._test_sparse_mask_shape(0, 0, [10, 10, 0], [2, 0], dtype, device, coalesced)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_zeros(self, device, dtype, coalesced):
        def _test_zeros(nnzs, shape, out_shape_i, out_shape_v=None):
            out_shape = out_shape_i + (out_shape_v or [])
            for nnz in nnzs:
                out, _, _ = self._gen_sparse(len(out_shape_i), nnz, out_shape, dtype, device, coalesced)
                torch.zeros(*shape, out=out, dtype=dtype, device=device)
                self.assertEqual(tuple(out.size()), tuple(shape))
                self.assertTrue(out._indices().numel() == out._values().numel() == 0)
                self.assertEqual(out._nnz(), 0)
                self.assertEqual(out.sparse_dim(), len(shape))
                self.assertEqual(out.dense_dim(), 0)

        def test_shape(i_shapes, v_shapes, shape, nnzs):
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    _test_zeros(nnzs, shape, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 4], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 0], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 0], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 0], [9, 12])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_zeros_like(self, device, dtype, coalesced):
        def _test_zeros_like(nnzs, template_shape_i, template_shape_v=None):
            template_shape_v = template_shape_v or []
            template_shape = template_shape_i + template_shape_v
            for nnz in nnzs:
                t, _, _ = self._gen_sparse(len(template_shape_i), nnz, template_shape, dtype, device, coalesced)
                res = torch.zeros_like(t)
                self.assertEqual(tuple(res.size()), tuple(template_shape))
                self.assertTrue(res._indices().numel() == res._values().numel() == 0)
                self.assertEqual(res._nnz(), 0)
                self.assertEqual(res.sparse_dim(), len(template_shape_i))
                self.assertEqual(res.dense_dim(), len(template_shape_v))

        def test_shape(i_shapes, v_shapes, nnzs):
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    _test_zeros_like(nnzs, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])

        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced)
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):

            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.zeros_like(x, memory_format=mem_format)

            result = torch.zeros_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        dense_tensor = sparse_tensor.to_dense()
        result = torch.zeros_like(dense_tensor, layout=torch.sparse_coo)
        self.assertEqual(dense_tensor.shape, result.shape)
        self.assertEqual(result.layout, torch.sparse_coo)

        sparse_zeros = torch.sparse_coo_tensor(dense_tensor.shape)
        self.assertEqual(result._indices().shape, sparse_zeros._indices().shape)
        self.assertEqual(result._values().shape, sparse_zeros._values().shape)

    def _assert_sparse_invars(self, t):
        # SparseTensor has the following invariants:
        # - sparse_dim + dense_dim = len(SparseTensor.shape)
        # - SparseTensor._indices().shape = (sparse_dim, nnz)
        # - SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
        self.assertEqual(t.sparse_dim() + t.dense_dim(), len(t.shape))
        self.assertEqual(tuple(t._indices().shape), (t.sparse_dim(), t._nnz()))
        self.assertEqual(tuple(t._values().shape), (t._nnz(), ) + t.shape[t.sparse_dim():])

    def _test_empty_like(self, sparse_tensor, dtype, device, coalesced):

        result = torch.empty_like(sparse_tensor)
        self.assertTrue(result.is_sparse)
        self._assert_sparse_invars(result)
        self.assertEqual(result.shape, sparse_tensor.shape)
        self.assertEqual(result.dtype, sparse_tensor.dtype)
        self.assertEqual(result.device, sparse_tensor.device)
        self.assertEqual(result.sparse_dim(), sparse_tensor.sparse_dim())
        self.assertEqual(result.dense_dim(), sparse_tensor.dense_dim())

        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced)
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):

            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.empty_like(x, memory_format=mem_format)

            result = torch.empty_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        with self.assertRaisesRegex(
            RuntimeError, r"Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA)' backend"
        ):
            dense_tensor = sparse_tensor.to_dense()
            result = torch.empty_like(dense_tensor, layout=torch.sparse_coo)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_empty_like(self, device, dtype, coalesced):
        # tests https://github.com/pytorch/pytorch/issues/43699

        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)

    def _test_narrow(self, input, narrow_args):
        expected = input.to_dense().narrow(*narrow_args)
        self.assertEqual(expected, input.narrow_copy(*narrow_args).to_dense())

    def _all_narrow_combs(self, shape):
        for dim, dim_sz in enumerate(shape):
            for start in range(dim_sz):
                for length in range(dim_sz - start):
                    yield [dim, start, length]

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_narrow(self, device, dtype, coalesced):
        shape = [3, 3, 4, 2]
        input, _, _ = self._gen_sparse(4, 19, shape, dtype, device, coalesced)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(input, narrow_args)

        self.assertRaises(RuntimeError, lambda: input.narrow_copy(-1, 0, 3))  # dim < 0
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(10, 0, 3))  # dim > input.dim()
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, shape[0] + 1, 3))  # start > size of dim
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, 2, shape[0]))  # start+length > size of dim

        with_dense, _, _ = self._gen_sparse(2, 7, shape, dtype, device, coalesced)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(with_dense, narrow_args)

        self.assertRaises(RuntimeError, lambda: with_dense.narrow_copy(10, 0, 3))  # dim > sparseDim + denseDim

    def _test_log1p_tensor(self, sparse_tensor, coalesced):
        def is_integral(dtype):
            return dtype in integral_types()

        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.log1p()
        is_integral_dtype = is_integral(sparse_tensor.dtype)
        self.assertEqual(expected_output, sparse_tensor.log1p().to_dense())
        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                sparse_tensor.coalesce().log1p_()
        else:
            self.assertEqual(expected_output, sparse_tensor.coalesce().log1p_().to_dense())

        if not coalesced:
            # test in-place op on uncoalesced input
            with self.assertRaisesRegex(RuntimeError, "log1p_ requires coalesced input"):
                sparse_tensor.log1p_()

        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "only Tensors of floating point dtype can require gradients"):
                sparse_tensor.requires_grad_()

    @coalescedonoff
    @dtypes(*all_types())
    def test_log1p(self, device, dtype, coalesced):
        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([3.0, 4.0, 5.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            ).coalesce()
            self._test_log1p_tensor(input_coalesced, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
                size=[4, 5, 2],
                device=device,
                dtype=dtype
            ).coalesce()
            self._test_log1p_tensor(input_coalesced, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            )
            self._test_log1p_tensor(input_uncoalesced, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                device=device,
                dtype=dtype
            )
            # empty tensors are coalesced at creation (nnz < 2) we must force the uncoalesced state
            input_uncoalesced._coalesced_(False)
            self._test_log1p_tensor(input_uncoalesced, coalesced)

    def _test_neg_negative(self, sparse_tensor):
        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.neg()

        ops = (
            torch.neg, torch.Tensor.neg, torch.Tensor.neg_,
            torch.negative, torch.Tensor.negative, torch.Tensor.negative_,
            operator.neg
        )
        for op in ops:
            sparse_tensor_copy = sparse_tensor.clone()
            self.assertEqual(expected_output, op(sparse_tensor_copy).to_dense())

            if op in (torch.neg, torch.negative):
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                op(sparse_tensor, out=sparse_tensor_out)
                self.assertEqual(expected_output, sparse_tensor_out.to_dense())

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_neg_negative(self, device, dtype, coalesced):

        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_neg_negative(input_coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_neg_negative(input_coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            self._test_neg_negative(input_uncoalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            self._test_neg_negative(input_uncoalesced)

    def _test_asin_arcsin(self, sparse_tensor, coalesced):
        def is_integral(dtype):
            return dtype in integral_types()
        is_integral_dtype = is_integral(sparse_tensor.dtype)

        dense_tensor = sparse_tensor.to_dense()
        expected_output = dense_tensor.asin()

        ops = (
            torch.asin, torch.Tensor.asin,
            torch.arcsin, torch.Tensor.arcsin,
        )
        for op in ops:
            self.assertEqual(expected_output, op(sparse_tensor).to_dense())
            if op in (torch.asin, torch.arcsin):
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                if not is_integral_dtype:
                    op(sparse_tensor, out=sparse_tensor_out)
                    self.assertEqual(expected_output, sparse_tensor_out.to_dense())
                else:
                    with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                        op(sparse_tensor, out=sparse_tensor_out)

        for op in (torch.Tensor.asin_, torch.Tensor.arcsin_):
            if is_integral_dtype:
                # test coalesce on integral dtype tensor
                with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                    op(sparse_tensor.clone().coalesce()).to_dense()
            else:
                self.assertEqual(expected_output, op(sparse_tensor.clone().coalesce()).to_dense())

            if not coalesced:
                # test in-place op on uncoalesced input
                with self.assertRaisesRegex(RuntimeError, "asin_ requires coalesced input"):
                    op(sparse_tensor)

    @coalescedonoff
    @dtypes(*all_types())
    def test_asin_arcsin(self, device, dtype, coalesced):
        if coalesced:
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2, 3]]),
                values=torch.tensor([0.5, -0.5, 0.7, -0.7]),
                size=[4, ],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_asin_arcsin(input_coalesced, coalesced)

            # hybrid sparse input
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-0.1, 0.24], [-0.44, 0.1]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            self._test_asin_arcsin(input_coalesced, coalesced)

        if not coalesced:
            # test uncoalesced input
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([0.3, -0.3, -0.4, 0.3, -0.5, 0.15]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            self._test_asin_arcsin(input_uncoalesced, coalesced)

            # test on empty sparse tensor
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            # empty tensors are coalesced at creation (nnz < 2) we must force the uncoalesced state
            input_uncoalesced._coalesced_(False)
            self._test_asin_arcsin(input_uncoalesced, coalesced)

    @coalescedonoff
    @dtypes(torch.double)
    def test_mv(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)
            t = torch.randn(dk, dtype=dtype, device=device)

            res = x.matmul(t)
            expected = self.safeToDense(x).matmul(t)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 1000, 20)
        test_shape(64, 10000, 10000, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 0, 0)
        test_shape(10, 100, 100, 0)
        test_shape(10, 100, 100, 20)

        with self.assertRaisesRegex(RuntimeError, r"mv: expected self\.size\(-1\) == vec\.size\(-1\)"):
            test_shape(10, 100, 10, 20)

        with self.assertRaisesRegex(RuntimeError, "mv: two tensor dim should be 2 and 1"):
            x, _, _ = self._gen_sparse(2, 20, [10, 100], dtype, device, coalesced)
            y, _, _ = self._gen_sparse(2, 20, [10, 100], dtype, device, coalesced)
            res = x.mv(y)

    @dtypes(*floating_and_complex_types())
    def test_sparse_add_coalesce(self, device, dtype):
        i = self.index_tensor([[1, 2, 1]], device=device)
        v = torch.tensor([3, 4, 5], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3]))
        y = self.sparse_tensor(i, v, torch.Size([3]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

        i = self.index_tensor([[1, 2, 1]], device=device)
        v = torch.empty([3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        y = self.sparse_tensor(i, v, torch.Size([3, 0]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

    @onlyCUDA
    def test_storage_not_null(self, device):
        x = torch.sparse_coo_tensor((2,), dtype=torch.float32, device=device)
        self.assertNotEqual(x.get_device(), -1)

        x = torch.sparse_coo_tensor((2, 0), dtype=torch.float32, device=device)
        self.assertNotEqual(x.get_device(), -1)

    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_same_gpu(self, devices):
        def check_device(x, device_id):
            self.assertEqual(x.get_device(), device_id)
            self.assertEqual(x._values().get_device(), device_id)
            self.assertEqual(x._indices().get_device(), device_id)

        dev1, dev2 = devices[0], devices[1]

        i = self.index_tensor([[2]], device=dev2)
        v = torch.tensor([5], device=dev2)
        x = self.sparse_tensor(i, v, torch.Size([3]), device=1)
        check_device(x, 1)

        i = self.index_tensor([[2]], device=dev2)
        v = torch.empty(1, 0, device=dev2)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]), device=1)
        check_device(x, 1)

        x = self.sparse_empty(3, device=1)
        check_device(x, 1)

        x = self.sparse_empty(3, 0, device=1)
        check_device(x, 1)

    def _test_new_device(self, size, device=torch.cuda):
        with torch.cuda.device(device):
            x = torch.sparse_coo_tensor(size, device='cuda', dtype=torch.float64)
        self.assertEqual(x.get_device(), device)
        x1 = x.new()
        x2 = x.new(2, 3)
        self.assertEqual(x1.get_device(), device)
        self.assertEqual(x2.get_device(), device)

    @onlyCUDA
    def test_new_device_single_gpu(self):
        self._test_new_device((), 0)
        self._test_new_device((30, 20), 0)
        self._test_new_device((30, 20, 10), 0)
        self._test_new_device((30, 20, 10, 0), 0)

    @onlyCUDA
    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_new_device_multi_gpu(self):
        self._test_new_device((), 1)
        self._test_new_device((30, 20), 1)
        self._test_new_device((30, 20, 10), 1)
        self._test_new_device((30, 20, 10, 0), 1)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_new(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            x, indices, values = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            if not x.is_cuda:
                # CUDA sparse tensors currently requires the size to be
                # specified if nDimV > 0
                out = x.new(indices, values).coalesce()
                x_c = x.coalesce()
                self.assertEqual((out.indices(), out.values()), (x_c.indices(), x_c.values()))
            self.assertEqual(x.new(indices, values, x.size()), x)

        test_shape(3, 10, 100)
        test_shape(3, 0, [100, 100, 0])

    @onlyCPU  # not really, but we only really want to run this once
    @dtypes(torch.float64, torch.float32, torch.float16, torch.cfloat, torch.cdouble)
    def test_factory(self, device, dtype):
        for test_empty_tensor in [True, False]:
            if test_empty_tensor:
                default_size = torch.Size([1, 3, 0])
                size = torch.Size([3, 3, 0])
            else:
                default_size = torch.Size([1, 3])
                size = torch.Size([3, 3])
            for include_size in [True, False]:
                for use_tensor_idx in [True, False]:
                    for use_tensor_val in [True, False]:
                        for use_cuda in ([False] if not torch.cuda.is_available() else [True, False]):
                            # have to include size with cuda sparse tensors
                            include_size = include_size or use_cuda
                            long_dtype = torch.int64
                            device = torch.device('cpu') if not use_cuda else \
                                torch.device(torch.cuda.device_count() - 1)
                            indices = torch.tensor(([0], [2]), dtype=long_dtype) if use_tensor_idx else ([0], [2])
                            if test_empty_tensor:
                                values = torch.empty(1, 0).to(dtype)
                            else:
                                if use_tensor_val:
                                    values = torch.tensor([1.], dtype=dtype)
                                else:
                                    values = 1.
                            if include_size:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, size, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            else:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            self.assertEqual(indices, sparse_tensor._indices())
                            self.assertEqual(values, sparse_tensor._values())
                            self.assertEqual(size if include_size else default_size, sparse_tensor.size())
                            self.assertEqual(dtype, sparse_tensor.dtype)
                            if use_cuda:
                                self.assertEqual(device, sparse_tensor._values().device)
                            self.assertEqual(True, sparse_tensor.requires_grad)

    @dtypes(torch.double, torch.cdouble)
    def test_factory_size_check(self, device, dtype):
        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.tensor([.5, .5], dtype=dtype, device=device)
        sizes = torch.Size([2, 3])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices.fill_(-1)
        with self.assertRaisesRegex(RuntimeError, "found negative index"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.empty([2, 1, 0], dtype=dtype, device=device)
        sizes = torch.Size([2, 3, 1, 0])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.empty([2, 2, 2], dtype=dtype, device=device)
        sizes = torch.Size([0, 0, 2, 2])
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=dtype, device=device)
        sizes = torch.Size([3, 3, 2])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        values = torch.empty([2, 1, 0], dtype=dtype, device=device)
        sizes = torch.Size([3, 3, 2, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

    def test_factory_empty_indices(self, device):
        tensor = torch.sparse_coo_tensor(torch.Size([2, 0]), device=device)
        expected_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0]), device=device)
        expected_indices = torch.empty((3, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0, 0]), device=device)
        expected_indices = torch.empty((4, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

    @dtypes(torch.double, torch.cdouble)
    def test_factory_nnz(self, device, dtype):
        indices = self.index_tensor([[0]], device=device)  # (sparse_dim, nnz): (1, 1)
        values = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)  # (nnz, ...): (2, 2)
        sizes = torch.Size([2, 2])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[0]], device=device)  # (sparse_dim, nnz): (1, 1)
        values = torch.empty([2, 0], dtype=dtype, device=device)  # (nnz, ...): (2, 0)
        sizes = torch.Size([2, 0])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

    @dtypes(torch.double, torch.cdouble)
    def test_factory_nnz_zero(self, device, dtype):
        def test_shape(i_shape, v_shape, size, expected_size):
            if size:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), torch.Size(size),
                                            dtype=dtype, device=device)
            else:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), dtype=dtype, device=device)
            expected_indices = torch.empty(i_shape, device=device, dtype=torch.int64)
            expected_values = torch.empty(v_shape, device=device, dtype=dtype)
            expected_size = torch.Size(expected_size)
            self.assertEqual(t._indices(), expected_indices)
            self.assertEqual(t._values(), expected_values)
            self.assertEqual(t.size(), expected_size)

        test_shape([1, 0], [0, 2, 4, 0], None, [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], None, [0, 0, 0, 2, 4, 0])
        test_shape([1, 0], [0, 2, 4, 0], [0, 2, 4, 0], [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [0, 0, 0, 2, 4, 0], [0, 0, 0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [1, 2, 3, 2, 4, 0], [1, 2, 3, 2, 4, 0])

    @dtypes(torch.double, torch.cdouble)
    def test_factory_dense_dim(self, device, dtype):
        indices = self.index_tensor([[0]], device=device)
        values = torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=dtype, device=device)
        sizes = torch.Size([1, 3, 4])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[0]], device=device)
        values = torch.empty([1, 2, 3, 0], dtype=dtype, device=device)
        sizes = torch.Size([1, 3, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

    @onlyCPU
    @dtypes(torch.float16, torch.float32, torch.float64, torch.cfloat, torch.cdouble, torch.int64)
    def test_factory_type_inference(self, device, dtype):
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=dtype))
        self.assertEqual(dtype, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1]))
        self.assertEqual(torch.int64, t.dtype)

        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.HalfTensor(1, 0))
        self.assertEqual(torch.float16, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.FloatTensor(1, 0))
        self.assertEqual(torch.float32, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.DoubleTensor(1, 0))
        self.assertEqual(torch.float64, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.LongTensor(1, 0))
        self.assertEqual(torch.int64, t.dtype)


    @onlyCUDA
    def test_factory_device_type_inference(self, device):
        # both indices/values are CUDA

        cpu_cuda = ('cpu', 'cuda')
        cpu_cuda_none = cpu_cuda + (None,)
        for indices_device, values_device, device in itertools.product(cpu_cuda,
                                                                       cpu_cuda,
                                                                       cpu_cuda_none):
            indices = torch.tensor(([0], [2]), device=indices_device)
            values = torch.tensor([1.], device=values_device)
            empty_values = torch.empty(1, 0).to(values_device)
            shape = (1, 3)
            empty_shape = (1, 3, 0)
            if device is None and indices_device != values_device:
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, values, shape, device=device)
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
            else:
                t = torch.sparse_coo_tensor(indices, values, shape, device=device)
                t_empty = torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
                should_be_cuda = (device == 'cuda' or (device is None and values_device == 'cuda'))
                self.assertEqual(should_be_cuda, t.is_cuda)
                self.assertEqual(t.is_cuda, t_empty.is_cuda)

    @onlyCPU
    def test_factory_copy(self, device):
        def test_tensor(indices, values, indices_equal, values_equal):
            sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=torch.float64, device=device)
            if indices_equal:
                self.assertEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            else:
                self.assertNotEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            if values_equal:
                self.assertEqual(values.data_ptr(), sparse_tensor._values().data_ptr())
            else:
                self.assertNotEqual(values.data_ptr(), sparse_tensor._values().data_ptr())

        # both correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, True, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, True, True)

        # only indices correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float16)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, True, True)  # An empty tensor's data_ptr is always equal to 0

        # only values correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, False, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, False, True)

        # neither correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, False, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, False, True)  # An empty tensor's data_ptr is always equal to 0

        # complex support
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = make_tensor([1, ], dtype=torch.cdouble, device=device)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = make_tensor([1, 1], dtype=torch.cdouble, device=device)
        test_tensor(indices, values, False, False)

    @onlyCPU  # just run once, we test both cpu and cuda
    def test_legacy_new_device(self, device):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])

        x = torch.sparse_coo_tensor(i, v, size, device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))

        if torch.cuda.is_available():
            x = torch.sparse_coo_tensor(i, v, size, device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))

    def test_legacy_new(self, device):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])
        s = torch.sparse_coo_tensor(i, v, size)

        self.assertEqual(torch.sparse_coo, s.new(device='cpu').layout)
        self.assertRaises(TypeError, lambda: s.new(v.untyped_storage()))
        self.assertRaises(TypeError, lambda: s.new(v))
        self.assertEqual(torch.sparse_coo, s.new(torch.Size([2, 3])).layout)
        self.assertRaises(TypeError, lambda: s.new([6]))

    @onlyCPU  # not really, but we only really want to run this once
    def test_dtypes(self, device):
        all_sparse_dtypes = all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16)
        do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cpu'))
        if torch.cuda.is_available():
            do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cuda:0'))

    def _test_empty_full(self, device, dtype, requires_grad):
        shape = (2, 3)
        layout = torch.sparse_coo

        def check_value(tensor, value=None, dtype=dtype, requires_grad=requires_grad):
            self.assertEqual(shape, tensor.shape)
            self.assertIs(dtype, tensor.dtype)
            self.assertIs(layout, tensor.layout)
            self.assertEqual(tensor.requires_grad, requires_grad)
            if tensor.is_cuda and device is not None:
                self.assertEqual(device, tensor.device)
            if value is not None:
                fill = tensor.empty(shape, dtype=dtype).fill_(value)
                self.assertEqual(tensor, fill)

        v = torch.sparse_coo_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        check_value(v)

        out = v.new()
        check_value(torch.zeros(shape, out=out, device=device, requires_grad=requires_grad))

        int64_dtype = torch.int64
        check_value(v.new_empty(shape), requires_grad=False)
        check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                    dtype=int64_dtype, requires_grad=False)
        check_value(torch.empty_like(v), requires_grad=False)
        check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                    dtype=int64_dtype, requires_grad=False)

    @onlyCPU  # not really, but we only really want to run this once
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize('requires_grad', (True, False))
    def test_empty_full(self, device, dtype, requires_grad):
        if requires_grad and not (dtype.is_floating_point or dtype.is_complex):
            self.skipTest(f'requires_grad==True requires float or complex dtype, got {dtype}')

        self._test_empty_full(device, dtype, requires_grad)
        if torch.cuda.is_available():
            self._test_empty_full(None, dtype, requires_grad)
            self._test_empty_full(torch.device('cuda:0'), dtype, requires_grad)

    def test_is_sparse(self, device):
        x = torch.randn(3, 3)
        self.assertFalse(x.is_sparse)

        x = torch.randn(3, 3, 0)
        self.assertFalse(x.is_sparse)

        x = self.sparse_empty(1, 0, device=device)
        self.assertTrue(x.is_sparse)

    def test_resize_as(self, device):
        def do_test(t):
            y = t.new().resize_as_(t).zero_()
            self.assertEqual(y.shape, t.shape)
            # Check that y can be added to t. Currently, this requires that
            # sparse_dim and dense_dim match.
            self.assertEqual(t, t + y)

        do_test(self.sparse_empty([3, 0], device=device))
        do_test(self.sparse_empty([3, 3], device=device))

    def _test_resize_shape(self, x_i, x_v, x_size, y_i, y_v, y_size, dtype, device):
        x_v_numel = torch.zeros(x_v).numel()
        y_v_numel = torch.zeros(y_v).numel()
        x = torch.sparse_coo_tensor(torch.zeros(x_i),
                                    torch.arange(x_v_numel).resize_(x_v).to(torch.float),
                                    torch.Size(x_size), dtype=dtype, device=device)
        x_dense = x.to_dense()
        y = torch.sparse_coo_tensor(torch.zeros(y_i),
                                    torch.ones(y_v).to(torch.float),
                                    torch.Size(y_size), dtype=dtype, device=device)
        y_dense = y.to_dense()
        x.resize_as_(y)
        x_dense.resize_as_(y_dense)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.sparse_dim(), y.sparse_dim())
        self.assertEqual(x.dense_dim(), y.dense_dim())
        self.assertEqual(x.shape, x_dense.shape)
        self.assertEqual(y.shape, y_dense.shape)
        # Here we make sure that the original data are preserved after resizing
        self.assertEqual(x.to_dense().view(-1)[0:x_v_numel].view(x_v),
                         x_dense.view(-1)[0:x_v_numel].view(x_v))

    @dtypes(torch.double, torch.cdouble)
    def test_resize(self, device, dtype):
        # 1. Expand the size of some dense dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 4], [2, 2, 4],
                                dtype=dtype, device=device)

        self._test_resize_shape([1, 1], [1, 2, 0], [2, 2, 0],
                                [1, 1], [1, 2, 4], [2, 2, 4],
                                dtype=dtype, device=device)

        # 2. Expand the size of some sparse dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 3], [4, 2, 3],
                                dtype=dtype, device=device)

        # 3. Change the shapes of both sparse and dense dimensions when nnz is zero [Supported]
        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 5], [1, 1, 2, 4, 5],
                                dtype=dtype, device=device)

        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 0], [1, 1, 2, 4, 0],
                                dtype=dtype, device=device)

        # 4. Add dims to dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 4], [2, 2, 3, 4],
                                    dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 0], [2, 2, 3, 0],
                                    dtype=dtype, device=device)

        # 5. Remove dims from dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2], [2, 2],
                                    dtype=dtype, device=device)

        # 6. Change the number of sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [2, 1], [1, 2, 3], [1, 2, 2, 3],
                                    dtype=dtype, device=device)

        # 7. Shrink the size of some sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3], [1, 2, 3],
                                    dtype=dtype, device=device)

        # 8. Shrink the size of some dense dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 2], [2, 2, 2],
                                    dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 0], [2, 2, 0],
                                    dtype=dtype, device=device)

    def test_is_nonzero(self, device):
        self.assertTrue(torch.sparse_coo_tensor(([0],), 1., (1,), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0., (1,), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0], [0]), 0., (1, 1), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (0., 0.), (1,), device=device).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (-1., 1.), (1,), device=device).is_nonzero())

        # scalar sparse tensor
        self.assertTrue(torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, [], device=device).is_nonzero())
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch.sparse_coo_tensor(([0, 1],), torch.empty(2, 0), (4, 0), device=device).is_nonzero()
        self.assertTrue(torch.sparse_coo_tensor(([0],), 2.3 - 4.5j, (1,), dtype=torch.cfloat, device=device)
                        .is_nonzero())
        self.assertTrue(torch.sparse_coo_tensor(([0],), 2.3 - 4.5j, (1,), dtype=torch.cdouble, device=device)
                        .is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0. + 0j, (1,), dtype=torch.cfloat, device=device)
                         .is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0. + 0j, (1,), dtype=torch.cdouble, device=device)
                         .is_nonzero())

    @dtypes(torch.double, torch.cdouble)
    def test_change_tensor_metadata(self, device, dtype):
        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]), dtype=dtype, device=device)
        i.resize_(2, 3)
        v.resize_(4, 5)
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.resize_as_(self.index_tensor([0, 1], device=device))
        v.resize_as_(torch.tensor([3, 4, 5], dtype=dtype, device=device))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.as_strided_((2, 1), (1, 1))
        v.as_strided_((1, 3), (1, 1))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.set_(self.index_tensor([0, 1], device=device))
        v.set_(torch.tensor([3, 4, 5], dtype=dtype, device=device))
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        i.transpose_(0, 1)
        v.transpose_(0, 1)
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

    @coalescedonoff
    @dtypes(torch.double)
    def test_pickle(self, device, dtype, coalesced):
        import pickle

        shape_sparse_dim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]

        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            indices_shape = torch.Size((sparse_dim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=device).view(indices_shape)
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if not coalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=dtype,
                                  device=device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.sparse_tensor(indices, values, shape)
            serialized = pickle.dumps(sp_tensor)
            sp_tensor_loaded = pickle.loads(serialized)
            self.assertEqual(sp_tensor, sp_tensor_loaded)

    def test_any(self, device):
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([False, False]), device=device)
        t_any = torch.tensor(False)
        self.assertEqual(torch.any(t), t_any)
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([True, False]), device=device)
        t_any = torch.tensor(True)
        self.assertEqual(torch.any(t), t_any)

    def test_isnan(self, device):
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([1, 4]), device=device)
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([False, False]), device=device)
        self.assertEqual(torch.isnan(t).int(), t_nan.int())
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([1, float("nan")]), device=device)
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([False, True]), device=device)
        self.assertEqual(torch.isnan(t).int(), t_nan.int())

    @coalescedonoff
    @dtypes(torch.float32, torch.float64)
    def test_div_rounding_mode(self, device, dtype, coalesced):
        sparse, _, _ = self._gen_sparse(2, 10, (10, 10), dtype,
                                        device, coalesced)
        dense = self.safeToDense(sparse)

        for mode in (None, 'floor', 'trunc'):
            actual = sparse.div(-2, rounding_mode=mode)
            expect = dense.div(-2, rounding_mode=mode)
            self.assertEqual(self.safeToDense(actual), expect)

            # Test inplace
            actual = sparse.clone().div_(-2, rounding_mode=mode)
            self.assertEqual(self.safeToDense(actual), expect)

            # Test out argument
            actual.zero_()
            torch.div(sparse, -2, rounding_mode=mode, out=actual)
            self.assertEqual(self.safeToDense(actual), expect)

    def test_div_by_sparse_error(self, device):
        self.assertRaisesRegex(RuntimeError, 'Sparse division requires',
                               lambda: torch.tensor(1., device=device).to_sparse()
                               / torch.tensor(1., device=device).to_sparse())

    def test_floor_divide_by_sparse_error(self, device):
        self.assertRaisesRegex(RuntimeError, 'Sparse floor division requires',
                               lambda: torch.tensor(1., device=device).to_sparse()
                               // torch.tensor(1., device=device).to_sparse())

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @onlyCPU
    def test_sparse_to_numpy(self, device):
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([1, 4]))
        self.assertRaises(TypeError, lambda: t.numpy())

    @coalescedonoff
    @dtypes(torch.double)
    def test_softmax(self, device, dtype, coalesced):
        import torch.nn.functional as F

        def to_dense(sparse, fill_value=None):
            """
            Return dense tensor from a sparse tensor using given fill value.
            """
            if fill_value is None or fill_value == 0:
                return sparse.to_dense()
            sparse = sparse.coalesce()
            dense = torch.full(sparse.shape, fill_value, dtype=sparse.dtype, device=sparse.device)
            for idx, value in zip(sparse._indices().t(), sparse._values()):
                dense[tuple(idx)] = value
            return dense

        def softmax_to_dense(sparse, dim):
            """Dense softmax of a sparse tensor. Useful only for testing softmax
            correctness.

            When computing softmax of a sparse tensor, the value of
            unspecified items is negative infinity rather than zero so
            that

              softmax(sparse.to_dense(fill_value=-inf), dim) == softmax(sparse, dim).to_dense()

            holds for non-empty lines. One empty lines, the softmax
            values are defined as 0 in order to preserve the sparsity
            of result.

            Note that in PyTorch, ``to_dense`` method does not
            implement the ``fill_value`` keyword argument.
            """
            dtype = sparse.dtype
            device = sparse.device
            dense = to_dense(sparse, fill_value=-float('inf'))
            r = F.softmax(dense, dim)
            # softmax on empty lines results nan, replace with zeros to match the definition
            r[r != r] = 0
            return r

        def sparse_softmax(sparse, dim):
            """Pure Python softmax of a sparse tensor. Assuming -inf for
            unspecified sparse tensor data. This is a prototype of
            sparse softmax algorithm in Python.
            """
            dtype = sparse.dtype
            device = sparse.device

            # softmax is non-linear operation, so sparse tensors must
            # be coalesced.
            sparse = sparse.coalesce()
            inf = float('inf')
            indices = sparse._indices()
            values = sparse._values()

            if dim < sparse.sparse_dim():
                nnz = sparse._nnz()

                # compute pool indices
                size = sparse.size()
                strides = torch.ones((sparse.sparse_dim(), 1), dtype=indices.dtype, device=indices.device)
                for i in reversed(range(sparse.sparse_dim() - 1)):
                    strides[i, 0] = strides[i + 1, 0] * size[i + 1]
                strides[dim, 0] = 0

                pool = (indices * strides).sum(dim=0)
                i2p = {}
                for i in range(nnz):
                    c = int(pool[i])
                    if c not in i2p:
                        i2p[c] = len(i2p)
                    pool[i] = i2p[c]

                # compute max
                dense_size = tuple(size[sparse.sparse_dim():])
                mx = torch.empty((pool.max() + 1,) + dense_size, dtype=dtype, device=device)
                mx[:] = -inf
                for n in range(nnz):
                    p = pool[n]
                    mx[p] = torch.max(mx[p], values[n])

                # apply exp to (v - mx) and sum the results
                exp_values = torch.empty_like(values)
                exp_sums = torch.zeros_like(mx)
                for n in range(nnz):
                    p = pool[n]
                    v = exp_values[n] = (values[n] - mx[p]).exp()
                    exp_sums[p] = exp_sums[p] + v

                # normalize with the sum of exponents
                for n in range(nnz):
                    p = pool[n]
                    exp_values[n] = exp_values[n] / exp_sums[p]

                return torch.sparse_coo_tensor(indices,
                                               exp_values,
                                               sparse.size(),
                                               dtype=dtype, device=device)

            elif dim < sparse.sparse_dim() + sparse.dense_dim():
                return torch.sparse_coo_tensor(indices,
                                               F.softmax(values, dim - sparse.sparse_dim() + 1),
                                               sparse.size(),
                                               dtype=dtype, device=device)
            else:
                raise ValueError(
                    '`dim(=%s)` must be smaller than `sparse_dim(=%s) + dense_dim(=%s)`'
                    % (dim, sparse.sparse_dim(), sparse.dense_dim()))

        def softmax_jacobian_analytic(x, dim):
            """Return Jacobian of softmax using analytic formula

               D_jS_i = S_i * (1[i==j] - S_j).

            where S = softmax(x, dim), x is dense tensor, i,j in
            range(x.shape[dim]).
            """
            y = F.softmax(x, dim)
            y[y != y] = 0  # replace nan-s with zeros
            J = torch.zeros((x.shape[dim],) + tuple(x.shape), dtype=x.dtype, device=x.device)
            si = [slice(None)] * len(y.shape)
            sj = [slice(None)] * len(y.shape)
            s = [slice(None)] * len(J.shape)
            for i in range(y.shape[dim]):
                si[dim] = i
                s[dim + 1] = i
                yi = y[tuple(si)]
                for j in range(y.shape[dim]):
                    sj[dim] = j
                    s[0] = j
                    if i == j:
                        J[tuple(s)] = yi * (1 - yi)
                    else:
                        yj = y[tuple(sj)]
                        J[tuple(s)] = - yi * yj
                    sj[dim] = slice(None)
                si[dim] = slice(None)
                s[dim + 1] = slice(None)
            return J

        def softmax_jacobian_autograd(x, dim, log=False):
            """Return Jacobian of softmax using PyTorch autograd feature.

            x can be dense or sparse tensor.
            """
            import itertools

            if x.is_sparse:
                x = x.coalesce()

            dtype = x.dtype
            device = x.device
            shape = tuple(x.shape)
            J = torch.zeros((shape[dim],) + shape, dtype=dtype, device=device)
            for i in range(shape[dim]):
                if x.is_sparse:
                    sparse_dim = x.sparse_dim()
                    dense_dim = x.dense_dim()
                    if dim < sparse_dim:
                        ranges = []
                        for j, sz in enumerate(shape[:sparse_dim]):
                            if dim == j:
                                ranges.append([i])
                            else:
                                ranges.append(list(range(sz)))
                        indices = torch.tensor(list(itertools.product(*ranges)), dtype=torch.long, device=device).t()
                        values = torch.ones((indices.shape[1],) + shape[sparse_dim:], dtype=dtype, device=device)
                    else:
                        ranges = []
                        for j, sz in enumerate(shape[:sparse_dim]):
                            ranges.append(list(range(sz)))
                        indices = torch.tensor(list(itertools.product(*ranges)), dtype=torch.long, device=device).t()
                        values = torch.zeros((indices.shape[1],) + shape[sparse_dim:], dtype=dtype, device=device)
                        sv = [slice(None)] * (dense_dim + 1)
                        sv[dim - sparse_dim + 1] = i
                        values[tuple(sv)] = 1
                    v = torch.sparse_coo_tensor(indices, values, shape, dtype=dtype, device=device)
                else:
                    v = torch.zeros_like(x)
                    sv = [slice(None)] * len(v.shape)
                    sv[dim] = i
                    v[tuple(sv)] = 1
                x_ = x.clone()
                x_.requires_grad_(True)

                if log:
                    if x_.is_sparse:
                        y = torch.sparse.log_softmax(x_, dim)
                    else:
                        y = F.log_softmax(x_, dim)
                else:
                    if x_.is_sparse:
                        y = torch.sparse.softmax(x_, dim)
                    else:
                        y = F.softmax(x_, dim)
                        # replace nan-s with zeros
                        y.data[y != y] = 0
                y.backward(v)
                g = x_.grad
                if not g.is_sparse:
                    # replace nan-s with zeros
                    g.data[g != g] = 0
                J[i] = g.to_dense() if g.is_sparse else g
            return J

        @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1166")
        def test_op(sparse_dims, nnz, with_size, coalesced):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims

            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)

            def sparse_log(x):
                return torch.sparse_coo_tensor(x._indices(), x._values().log(),
                                               x.size(), dtype=x.dtype, device=x.device)

            # Check dim out of bounds
            with self.assertRaisesRegex(IndexError, r"Dimension out of range"):
                torch.sparse.softmax(x, x.dim())
            with self.assertRaisesRegex(IndexError, r"Dimension out of range"):
                torch.sparse.softmax(x, -x.dim() - 1)

            for dim in range(x.dim()):
                # Check sparse softmax definition

                # check Python sparse softmax
                y = sparse_softmax(x, dim)
                r1 = softmax_to_dense(x, dim)
                r2 = y.to_dense()
                self.assertEqual(r1, r2)

                # check C++ sparse softmax
                for d in (dim, dim - x.dim()):
                    y1 = torch.sparse.softmax(x, d)
                    self.assertEqual(y, y1)

                    # check C++ sparse log_softmax
                    ly1 = torch.sparse.log_softmax(x, d)
                    self.assertEqual(ly1, sparse_log(y1))

                # Check autograd support on sparse softmax

                # check softmax Jacobian definition for dense input
                x1 = to_dense(x, fill_value=float('-inf'))
                J = softmax_jacobian_analytic(x1, dim)
                assert J.shape[0] == x.shape[dim]
                assert J.shape[dim + 1] == x.shape[dim]

                # check softmax Jacobian from autograd, dense input
                J2 = softmax_jacobian_autograd(x1, dim)
                self.assertEqual(J, J2)

                # check softmax Jacobian from autograd, sparse input
                J3 = softmax_jacobian_autograd(x, dim)
                self.assertEqual(J, J3)

                '''
                y = softmax(x, dim)
                z = log(y) = log_softmax(x, dim)
                Dy/Dx = J
                Dz/Dx = Dz/Dy Dy/Dx = 1/y * J
                => J = J_log * y
                '''
                # log_softmax Jacobian from autograd, dense input
                J2_log = softmax_jacobian_autograd(x1, dim, log=True)

                # log_softmax Jacobian from autograd, sparse input
                J3_log = softmax_jacobian_autograd(x, dim, log=True)

                J = J.transpose(0, dim + 1)
                J2_log = J2_log.transpose(0, dim + 1)
                J3_log = J3_log.transpose(0, dim + 1)
                self.assertEqual(J, J2_log * r1)
                self.assertEqual(J, J3_log * r1)

                if dim == 0:
                    # check dtype argument
                    other_dtype = torch.float32
                    y2 = torch.sparse.softmax(x, dim, dtype=other_dtype)
                    self.assertEqual(y2.dtype, other_dtype)
                    self.assertEqual(y2, y1.type(other_dtype))

                    ly2 = torch.sparse.log_softmax(x, dim, dtype=other_dtype)
                    self.assertEqual(ly2.dtype, other_dtype)
                    self.assertEqual(ly2, ly1.type(other_dtype))

        test_op(1, 10, [3], coalesced)
        test_op(1, 10, [2, 3], coalesced)
        test_op(1, 10, [3, 2], coalesced)
        test_op(2, 10, [2, 3, 4], coalesced)
        test_op(2, 10, [3, 4], coalesced)
        test_op(2, 5, [5, 4], coalesced)
        test_op(2, 10, [3, 4, 2], coalesced)
        test_op(3, 10, [3, 4, 2], coalesced)
        test_op(3, 100, [3, 4, 2], coalesced)
        test_op(3, 100, [3, 4, 2, 3], coalesced)
        test_op(3, 100, [3, 4, 2, 3, 5, 2], coalesced)
        test_op(4, 100, [3, 4, 2, 3, 5, 2], coalesced)


    def _check_zero_nnz_softmax_op(self, func, ndim, device, dtype):
        # create a sparse tensor with shape (0,..., 3) it has no materialize values
        t = torch.sparse_coo_tensor([[] for _ in range(ndim)], [], (0,) * (ndim - 1) + (3,), device=device, dtype=dtype)
        out = func(t, 0)
        self.assertEqual(out, torch.zeros_like(t))

        # gradient
        t = t.requires_grad_()
        gradcheck(lambda x: func(x, 0).to_dense(), (t,), masked=True)


    @dtypes(torch.double, torch.float)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    def test_softmax_zero_nnz(self, device, dtype):
        self._check_zero_nnz_softmax_op(torch.sparse.softmax, 1, device, dtype)
        self._check_zero_nnz_softmax_op(torch.sparse.softmax, 10, device, dtype)

    @dtypes(torch.double, torch.float)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    def test_log_softmax_zero_nnz(self, device, dtype):
        self._check_zero_nnz_softmax_op(torch.sparse.log_softmax, 1, device, dtype)
        self._check_zero_nnz_softmax_op(torch.sparse.log_softmax, 10, device, dtype)

    # TODO: Check after why ROCm's cusparseXcsrgemm2Nnz function doesn't return the same nnz value as CUDA
    @skipIfRocm
    @coalescedonoff
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_types_and(*[torch.half] if SM53OrLater else [],
                                      *[torch.bfloat16] if SM80OrLater else [],
                                      torch.complex64,
                                      *[torch.complex128] if CUSPARSE_SPMM_COMPLEX128_SUPPORTED else []))
    @unittest.skipIf(TEST_WITH_CROSSREF, "not working with fake tensor")
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2, torch.complex64: 1e-2, torch.float32: 1e-2})
    def test_sparse_matmul(self, device, dtype, coalesced):
        """
        This function test `torch.sparse.mm` when both the mat1 and mat2 are sparse tensors.
        """

        def ref_sparse_mm(a, b):
            return a.to_dense() @ b.to_dense()

        def grad_with_custom_sparsity_pattern_test_helper(sparse_dims, nnz, shape_a, shape_b):
            def test_grad_dense(a_s, b_s, g_s):
                a = a_s.to_dense().detach()
                b = b_s.to_dense().detach()
                g = g_s.to_dense().detach()

                a.requires_grad_(True)
                b.requires_grad_(True)
                c = a @ b
                c.backward(g)
                return a.grad.sparse_mask(a_s.coalesce()), b.grad.sparse_mask(b_s.coalesce())

            a, _, _ = self._gen_sparse(sparse_dims, nnz, shape_a, dtype, device, coalesced)
            b, _, _ = self._gen_sparse(sparse_dims, nnz, shape_b, dtype, device, coalesced)
            a.requires_grad_(True)
            b.requires_grad_(True)

            c = torch.sparse.mm(a, b)
            c2 = c.to_dense().detach()
            c2 = torch.rand_like(c2)
            g = c2.sparse_mask(c.coalesce())

            c.backward(g)

            a_grad, b_grad = test_grad_dense(a, b, g)

            # We convert grad to dense since dense and sparse mm
            # implementations handle materialized zeroes differently.
            self.assertEqual(a.grad.to_dense(), a_grad.to_dense())
            self.assertEqual(b.grad.to_dense(), b_grad.to_dense())

        def test_sparse_matmul(sparse_dims, nnz, shape_a, shape_b):
            a, i_a, v_a = self._gen_sparse(sparse_dims, nnz, shape_a, dtype, device, coalesced)
            b, i_b, v_b = self._gen_sparse(sparse_dims, nnz, shape_b, dtype, device, coalesced)

            # dense implementation
            r1 = ref_sparse_mm(a, b)

            # cpp implementation
            r2 = torch.sparse.mm(a, b)
            self.assertEqual(r1, r2.to_dense())

            # Check result is truly coalesced
            self.assertTrue(r2.is_coalesced() and is_coalesced_indices(r2))

            if dtype in [torch.double, torch.cdouble]:
                a.requires_grad_(True)
                b.requires_grad_(True)

                # check autograd support on sparse matmul
                def fn(D1, D2):
                    return torch.sparse.mm(D1, D2).to_dense()

                if a.is_cuda:
                    # For cuda, `nondet_tol` is set with `1e-5`
                    # This is because cuSparse sometimes returns approximate zero values like `~e-323`
                    # TODO: Check this cuSparse issue.
                    # This happens when you do chain multiplication `torch.sparse.mm` operations
                    gradcheck(fn, (a, b), nondet_tol=1e-5, masked=True)
                else:
                    gradcheck(fn, (a, b), masked=True)
                grad_with_custom_sparsity_pattern_test_helper(sparse_dims, nnz, shape_a, shape_b)

        def test_error_cases():
            def fn(sparse_dims, nnz, shape_a, shape_b):
                a, i_a, v_a = self._gen_sparse(sparse_dims, nnz, shape_a, dtype, device, coalesced)
                b, i_b, v_b = self._gen_sparse(sparse_dims, nnz, shape_b, dtype, device, coalesced)
                r2 = torch.sparse.mm(a, b)

            # This is not a matrix
            self.assertRaises(RuntimeError, lambda: fn(3, 4, [2, 2, 2], [2, 2, 2]))

            # Shapes does not
            self.assertRaisesRegex(RuntimeError,
                                   r"mat1 and mat2 shapes cannot be multiplied \(2x3 and 4x2\)",
                                   lambda: fn(2, 10, [2, 3], [4, 2]))

            def different_dtypes():
                a, i_a, v_a = self._gen_sparse(2, 10, [2, 2], dtype, device, coalesced)
                b, i_b, v_b = self._gen_sparse(2, 10, [2, 2], dtype, device, coalesced)
                r2 = torch.sparse.mm(a.to(torch.float64), a.to(torch.float32))

            self.assertRaisesRegex(RuntimeError, 'mat1 dtype Double does not match mat2 dtype Float', different_dtypes)

        for n in range(2, 5):
            for m in range(2, 8):
                for p in range(2, 8):
                    test_sparse_matmul(2, 10, [n, m], [m, p])

        test_sparse_matmul(2, 0, [0, 0], [0, 0])
        test_sparse_matmul(2, 0, [0, 10], [10, 0])
        test_error_cases()

    @coalescedonoff
    @dtypes(torch.double)
    def test_assign(self, device, dtype, coalesced):
        def assign_to():
            a, i_a, v_a = self._gen_sparse(2, 5, [2, 3], dtype, device, coalesced)
            a[0] = 100

        self.assertRaises(TypeError, assign_to)

    @dtypes(torch.double, torch.cdouble)
    def test_full_broadcast_to(self, device, dtype):
        def can_broadcast(s0, s1):
            s0 = tuple(reversed(s0))
            s1 = tuple(reversed(s1))
            for i in range(len(s0)):
                if s0[i] != 1 and s0[i] != s1[i]:
                    return False
            return True
        sizes = (
            (), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)
        )
        for s0, s1 in itertools.combinations(sizes, r=2):
            t = make_tensor(s0, dtype=dtype, device=device, low=-9, high=9)
            for sparse_dims in range(1, len(s0) + 1):
                s = t.to_sparse(sparse_dims)
                if can_broadcast(s0, s1):
                    t_res = torch.broadcast_to(t, s1)
                    s_res = torch._sparse_broadcast_to(s, s1)
                    torch._validate_sparse_coo_tensor_args(s_res._indices(), s_res._values(), s_res.shape)
                    if s_res.is_coalesced():
                        # ensure that is_coalesced is estimated correctly
                        self.assertEqual(s_res, torch.sparse_coo_tensor(s_res._indices(), s_res._values(), s_res.shape).coalesce())
                    self.assertEqual(s_res.to_dense(), t_res)
                else:
                    with self.assertRaisesRegex(RuntimeError,
                                                r"The expanded size of the tensor \(\d\) "
                                                r"must match the existing size \(\d\)"):
                        torch._sparse_broadcast_to(s, s1)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_sparse_broadcast_to(self, device, dtype, coalesced):
        def test(sparse_dims, nnz, with_size, new_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            y = self.safeToDense(x)
            x1 = torch._sparse_broadcast_to(x, new_size)
            y1 = y.broadcast_to(new_size)
            self.assertEqual(self.safeToDense(x1), y1)

        test(4, 6, [7, 3, 1, 3, 0], [7, 3, 4, 3, 0])
        test(4, 6, [7, 3, 1, 3, 0], [2, 7, 3, 1, 3, 0])
        test(4, 6, [7, 3, 1, 3, 1, 3], [7, 3, 1, 3, 2, 3])
        test(4, 6, [7, 3, 1, 3, 2, 1], [7, 3, 1, 3, 2, 3])

    def _test_mul_skips(self, device, dtype, coalesced):
        skipTestIfUncoalesced = False
        # This case always coalesce inputs and that could lead to loss of precision,
        # hence it is inhibited for float16/bfloat16 by providing already coalesced tensors.
        if not coalesced and dtype in {torch.float16, torch.bfloat16}:
            skipTestIfUncoalesced = True
        # to_dense is problematic for boolean non-coalesced CUDA tensors
        # see https://github.com/pytorch/pytorch/issues/81648
        if not coalesced and dtype == torch.bool and torch.device(device).type == "cuda":
            skipTestIfUncoalesced = True

        if skipTestIfUncoalesced:
            self.skipTest(f"Test with dtype={dtype}, device={device} runs only with coalesced inputs")

    @coalescedonoff
    # NOTE: addcmul_out is not implemented for bool.
    @dtypes(*all_types_and_complex_and(torch.bfloat16, torch.float16))
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_sparse_sparse_mul(self, device, dtype, coalesced):
        self._test_mul_skips(device, dtype, coalesced)

        shape = (2, 3, 4, 10)
        nnz = 10

        def check(self, x, y):
            res_sparse = x * y
            res_dense = x.to_dense() * y.to_dense()
            self.assertEqual(res_sparse.to_dense(), res_dense)

        def check_empty(sparse_shape, nnz, dense_shape, coalesce):
            from itertools import product
            for nnz_val, shape_suffix in product((nnz, 0), ((), (0,))):
                empty_sparse_shape = sparse_shape + shape_suffix
                empty_dense_shape = dense_shape + shape_suffix
                x = self._gen_sparse(sparse_dim, nnz_val, empty_sparse_shape, dtype, device, coalesce)[0]
                check(self, x, x)

        # TODO: uncomment once backward is implemented for sparse tensors that broadcast in dense dims.
        # def check_autograd(x, y):
        #     if dtype in {torch.double, torch.cdouble}:
        #         xa = x.detach().clone().requires_grad_(True)
        #         ya = y.detach().clone().requires_grad_(True)
        #         gradcheck(lambda a, b: (a * b).to_dense(), (xa, ya), masked=True)
        #         gradcheck(lambda a, b: (a * b).to_dense(), (ya, xa), masked=True)

        for dim in range(len(shape) + 1):
            sub_shape = shape[dim:]
            sparse_dim = len(sub_shape) // 2

            check_empty(sub_shape, nnz, shape, coalesced)

            x = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            y = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            check(self, x, y)
            # TODO: uncomment once supported
            # check_autograd(x, y)

            # check broadcasting in dense dims
            for d in range(sparse_dim, len(sub_shape)):
                new_shape = sub_shape[:d] + (1,) + sub_shape[d + 1:]
                y = self._gen_sparse(sparse_dim, nnz, new_shape, dtype, device, coalesced)[0]
                check(self, x, y)
                # TODO: uncomment once supported
                # check_autograd(x, y)

    @coalescedonoff
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_sparse_dense_mul(self, device, dtype, coalesced):
        self._test_mul_skips(device, dtype, coalesced)

        shape = (2, 3, 4, 10)
        nnz = 10

        def check(self, s, d):
            res = d * s

            # check commutativity
            self.assertEqual(res, s * d)

            # check correctness
            self.assertEqual(res.to_dense(), s.to_dense() * d)

            # check in-placeness for dense
            if d.dim() >= s.dim():
                dc = d.clone()
                self.assertEqual(d.mul_(s), dc.mul_(s.to_dense()))

            # check in-placeness for sparse
            if s.dim() >= d.dim():
                # for sparse
                sc = s.clone()
                self.assertEqual(s.mul_(d).to_dense(), sc.to_dense().mul_(d))

        for dim in range(len(shape) + 1):
            sub_shape = shape[dim:]
            sparse_dim = len(sub_shape) // 2

            def check_empty(sparse_shape, nnz, dense_shape, coalesce):
                from itertools import product
                for nnz_val, shape_suffix in product((nnz, 0), ((), (0,))):
                    empty_sparse_shape = sparse_shape + shape_suffix
                    empty_dense_shape = dense_shape + shape_suffix
                    s = self._gen_sparse(sparse_dim, nnz_val, empty_sparse_shape, dtype, device, coalesce)[0]
                    d = make_tensor(empty_dense_shape, dtype=dtype, device=device)
                    check(self, s, d)

            # check scalar multiplication
            s = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            for scalar in (True, 1, 1.0):
                res_sparse_right = s * scalar
                res_sparse_left = scalar * s
                res_dense = s.to_dense() * scalar
                # check correctness and dtype
                self.assertEqual(s.to(res_sparse_right.dtype), res_sparse_right)
                self.assertEqual(res_sparse_right, res_sparse_left)
                self.assertEqual(res_sparse_right.dtype, res_dense.dtype)
                self.assertEqual(res_sparse_left.dtype, res_dense.dtype)
                # check scalar as 0-dim sparse tensor
                tscalar = torch.tensor(scalar, device=device)
                sscalar = tscalar.to_sparse()
                res_sparse_right = s * sscalar
                res_sparse_left = sscalar * s
                self.assertEqual(res_sparse_right, res_sparse_left)
                self.assertEqual(s.to(res_sparse_right.dtype), res_sparse_right)

            # check non-coalesced 0-dim scalar
            # we skip torch.bool because for such tensors
            # coalesce.to_dense != to_dense
            if dtype == torch.bool:
                return

            for scalar_dtype in (int, float):
                scalar = scalar_dtype(1)
                idx = torch.tensor([], device=device).reshape(0, 2)
                val = torch.tensor([scalar, scalar], device=device)
                sscalar = torch.sparse_coo_tensor(idx, val, ())
                res_dense = s.to_dense() * sscalar.to_dense()
                self.assertEqual((s * sscalar).to_dense(), res_dense)
                self.assertEqual((sscalar * s).to_dense(), res_dense)

            # Case 1: sparse broadcasts over dense
            s = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            d = make_tensor(shape, dtype=dtype, device=device)
            check(self, s, d)
            check_empty(sub_shape, nnz, shape, coalesced)

            # Case 2: dense broadcasts over sparse
            s = self._gen_sparse(3, nnz, shape, dtype, device, coalesced)[0]
            d = make_tensor(sub_shape, dtype=dtype, device=device)
            check(self, s, d)
            check_empty(shape, nnz, sub_shape, coalesced)

    @unittest.skipIf(not TEST_NUMPY, "NumPy is not available")
    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.bool))
    def test_sparse_spdiags(self, device, dtype):

        make_diags = functools.partial(make_tensor, dtype=dtype, device=device)
        make_offsets = functools.partial(torch.tensor, dtype=torch.long, device=device)

        if TEST_SCIPY:
            def reference(diags, offsets, shape):
                return scipy.sparse.spdiags(diags, offsets, *shape).toarray()

        else:
            def reference(diags, offsets, shape):
                result = torch.zeros(shape, dtype=dtype, device=device)
                for i, off in enumerate(offsets):
                    res_view = result.diagonal(off)
                    data = diags[i]
                    if off > 0:
                        data = data[off:]

                    m = min(res_view.shape[0], data.shape[0])
                    res_view[:m] = data[:m]
                return result

        def check_valid(diags, offsets, shape, layout=None):
            ref_out = reference(diags, offsets, shape)
            out = torch.sparse.spdiags(diags, offsets, shape, layout=layout)
            if layout is None:
                ex_layout = torch.sparse_coo
            else:
                ex_layout = layout
            out_dense = out.to_dense()
            self.assertTrue(out.layout == ex_layout, f"Output layout {out.layout} expected {ex_layout}")
            self.assertEqual(out_dense, ref_out, f"Result:\n{out_dense} does not match reference:\n{ref_out}")

        def check_invalid(args, error):
            with self.assertRaisesRegex(RuntimeError, error):
                torch.sparse.spdiags(*args)

        def valid_cases():
            # some normal cases
            yield (make_diags((1, 5)), make_offsets([0]), (5, 5))
            yield (make_diags((3, 3)), make_offsets([-1, 0, 1]), (4, 4))
            # noncontigous diags
            yield (make_diags((5, 4), noncontiguous=True), make_offsets([-1, 1, 0, 2, -2]), (5, 5))
            # noncontigous offsets
            yield (make_diags((3, 4)), make_offsets([1, -1, 0, -2, 2])[::2], (5, 5))
            # noncontigous diags + offsets
            yield (make_diags((3, 4), noncontiguous=True), make_offsets([1, -1, 0, -2, 2])[::2], (5, 5))
            # correct dimensionality, 2d, 2d , and shapes match, but the number of diagonals is zero
            yield (make_diags((0, 3)), make_offsets([]), (3, 3))
            # forward rotation of upper diagonals
            yield (make_diags((3, 8)), make_offsets([1, 2, 3]), (4, 4))
            # rotation exausts input space to read from
            yield (make_diags((2, 3)), make_offsets([2, 1]), (3, 3))
            # Simple cases repeated with special output format
            yield (make_diags((1, 5)), make_offsets([0]), (5, 5), torch.sparse_csc)
            yield (make_diags((3, 3)), make_offsets([-1, 0, 1]), (4, 4), torch.sparse_csr)
            # vector diags
            yield (make_diags((3, )), make_offsets([1]), (4, 4))
            # Scalar offset
            yield (make_diags((1, 3)), make_offsets(2), (4, 4))
            # offsets out of range
            yield (make_diags((1, 3)), make_offsets([3]), (3, 3))
            yield (make_diags((1, 3)), make_offsets([-3]), (3, 3))

        for case in valid_cases():
            check_valid(*case)

        def invalid_cases():
            yield (make_diags((1, 3)), make_offsets([0]), (3, 2, 3)), "Output shape must be 2d"
            yield (make_diags((2, 3)), make_offsets([[1, 2], [0, 3]]), (3, 3)), "Offsets must be scalar or vector"
            yield (make_diags((3, 2, 3)), make_offsets([0, 1, 2]), (4, 4)), "Diagonals must be vector or matrix"
            yield (make_diags((3, 3)), make_offsets([-1, 0]), (3, 3)),\
                r"Number of diagonals \(\d\) does not match the number of offsets \(\d\)"
            yield (make_diags((5,)), make_offsets([0, 1, 2, 3, 4]), (3, 3)),\
                r"Number of diagonals \(\d\) does not match the number of offsets \(\d\)"
            yield (make_diags((2, 2)), make_offsets([-1, 0]), (2, 3), torch.strided),\
                r"Only output layouts \(\w+, \w+, \w+\) are supported, got \w+"
            yield (make_diags((2, 5)), make_offsets([0, 0]), (5, 5)), "Offset tensor contains duplicate values"
            yield (make_diags((1, 5)), make_offsets([0]).to(torch.int32), (5, 5)), r"Offset Tensor must have dtype Long but got \w+"


        for case, error_regex in invalid_cases():
            check_invalid(case, error_regex)

    def test_small_nnz_coalesced(self):
        # creating a coo tensor with nnz == 0 is always coalesced
        self.assertTrue(torch.sparse_coo_tensor([[], []], [], (2, 2)).is_coalesced())
        # same for a coo tensor with only 1 nnz
        self.assertTrue(torch.sparse_coo_tensor([[0], [0]], [1], (2, 2)).is_coalesced())
        # two or more nnz coalesced is false as it can't be verified without an expensive check
        self.assertFalse(torch.sparse_coo_tensor([[0, 0], [0, 0]], [1, 2], (2, 2)).is_coalesced())
        # even if there are no duplicates
        self.assertFalse(torch.sparse_coo_tensor([[0, 1], [0, 1]], [1, 2], (2, 2)).is_coalesced())

    @coalescedonoff
    @dtypes(*all_types_and_complex_and(torch.bool))
    def test_sum(self, device, dtype, coalesced):
        def run_test(shape, nnz):
            a = self._gen_sparse(2, nnz, shape, dtype, device, coalesced)[0]
            self.assertEqual(a.sum(), a._values().sum())
            if dtype.is_floating_point or dtype.is_complex:
                a.requires_grad_(True)
                a_inter = a.sum()
                a_inter.abs().backward()
                with torch.no_grad():
                    self.assertEqual(a.grad, torch.ones(shape, dtype=dtype, device=device) * torch.sgn(a_inter))
        for shape in [(10, 5), (10, 10)]:
            run_test(shape, 0)
            run_test(shape, max(shape))
            run_test(shape, shape[0] * shape[1])


class TestSparseOneOff(TestCase):
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_from_cpu(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"):
            torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                    torch.randn(4, 4, 4),
                                    [3, 4, 4])

        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"):
            torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                    torch.randn(4, 4, 4, 0),
                                    [3, 4, 4, 0])

        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"):
            torch.sparse_coo_tensor(torch.empty(1, 0).long().cuda(),
                                    torch.randn(0, 4, 4, 0),
                                    [0, 4, 4, 0])

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_sparse_cpu_dense_add(self):
        x = torch.zeros(3, 4, 4)
        sparse_y = torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                           torch.randn(4, 4, 4).cuda(),
                                           [3, 4, 4])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y

        x = torch.zeros(3, 4, 4, 0)
        sparse_y = torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                           torch.randn(4, 4, 4, 0).cuda(),
                                           [3, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y

        x = torch.zeros(0, 4, 4, 0)
        sparse_y = torch.sparse_coo_tensor(torch.empty(1, 0).long().cuda(),
                                           torch.randn(0, 4, 4, 0).cuda(),
                                           [0, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            x + sparse_y


def _sparse_to_dense(tensor):
    if tensor.dtype != torch.bool:
        return tensor.to_dense(masked_grad=True)

    # to_dense uses coalesce which isn't implemented for bool
    return tensor.to(torch.int8).to_dense().to(torch.bool)


_sparse_unary_ops = ops(sparse_unary_ufuncs, dtypes=OpDTypes.supported,
                        allowed_dtypes=all_types_and_complex())
class TestSparseUnaryUfuncs(TestCase):
    exact_dtype = True


    @_sparse_unary_ops
    def test_sparse_consistency(self, device, dtype, op):
        sample = first_sample(self, op.sample_inputs(device, dtype))
        assert isinstance(sample.input, torch.Tensor)

        expected = op(sample.input, *sample.args, **sample.kwargs)
        assert torch.is_tensor(expected)
        output = op(sample.input.to_sparse(), *sample.args, **sample.kwargs)
        assert torch.is_tensor(output)
        self.assertEqual(_sparse_to_dense(output), expected)

    @_sparse_unary_ops
    def test_out(self, device, dtype, op):
        if not op.supports_out:
            self.skipTest("Skipped! Out not supported")

        sample = first_sample(self, op.sample_inputs(device, dtype))
        sample.input = sample.input.to_sparse()
        expect = op(sample.input, *sample.args, **sample.kwargs)

        out = torch.sparse_coo_tensor(sample.input.shape, device=device,
                                      dtype=expect.dtype)
        op(sample.input, *sample.args, **sample.kwargs, out=out)
        self.assertEqual(out, expect)

    @_sparse_unary_ops
    def test_inplace(self, device, dtype, op):
        if op.inplace_variant is None:
            self.skipTest("Skipped! Out not supported")

        sample = first_sample(self, op.sample_inputs(device, dtype))
        sample.input = sample.input.to_sparse().coalesce()
        expect = op(sample.input, *sample.args, **sample.kwargs)

        if not torch.can_cast(expect.dtype, dtype):
            with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
            return

        actual = op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
        self.assertIs(actual, sample.input)
        self.assertEqual(actual, expect)

    @_sparse_unary_ops
    def test_sparse_zero_dims(self, device, dtype, op):
        # test 0x0 sparse_coo_tensor
        indices = torch.empty(2, 0, dtype=torch.int64)
        values = torch.empty(0, dtype=dtype)
        sparse_0x0 = torch.sparse_coo_tensor(indices, values, (0, 0))
        expected = torch.sparse_coo_tensor(indices, op(values), (0, 0))
        actual = op(sparse_0x0)
        self.assertEqual(expected, actual)

    @_sparse_unary_ops
    def test_sparse_zeros(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)

        zero_input = torch.zeros((), device=device, dtype=dtype)
        sparse_input = torch.sparse_coo_tensor((), dtype=dtype, device=device)

        expect = op(zero_input)
        actual = op(sparse_input)
        self.assertEqual(expect, _sparse_to_dense(actual))

    @ops(sparse_unary_ufuncs, dtypes=OpDTypes.supported,
         allowed_dtypes=[torch.double, torch.cdouble])
    def test_sparse_fn_grad(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Op doesn't support autograd")

        for sample in op.sample_inputs(device, dtype):
            sparse_input = sample.input.to_sparse().detach().requires_grad_(True)

            def fn(x):
                return _sparse_to_dense(
                    op(x, *sample.args, **sample.kwargs))

            self.assertTrue(gradcheck(
                fn,
                (sparse_input,),
                check_batched_grad=False,
                check_grad_dtypes=True,
                nondet_tol=op.gradcheck_nondet_tol,
                fast_mode=op.gradcheck_fast_mode,
                masked=True))


class TestSparseMaskedReductions(TestCase):
    exact_dtype = True

    @ops(sparse_masked_reduction_ops)
    def test_future_empty_dim(self, device, dtype, op):
        """Currently, `dim=()` in reductions operations means "reduce over
        all dimensions" while in future, it will read "no reduce". See
        https://github.com/pytorch/pytorch/issues/29137

        For sparse masked reductions, we'll implement the current behavior.

        For testing, we'll use samples with `dim=0` and map it to
        `dim=()` until
        torch.testing._internal.common_methods_invocations._generate_reduction_kwargs
        is made to generate samples with `dim=()` for non-scalar
        inputs. With this and after gh-29137 is resolved, this test
        can be deleted. See also `torch.masked._canonical_dim`
        implementation about changing the `dim=()` behavior.
        """

        samples = op.sample_inputs_func(op, device, dtype, requires_grad=False)
        op_name = op.name.replace('masked.', '')
        for sample_input in samples:
            if sample_input.kwargs.get('dim') != 0:
                continue
            sample_input_kwargs = dict(sample_input.kwargs)
            sample_input_kwargs['dim'] = ()    # reduce over all dimensions

            t = sample_input.input
            mask = sample_input_kwargs.get('mask')
            if mask is None and op_name in {'prod', 'amax', 'amin'}:
                # FIXME: for now reductions with non-zero reduction identity and
                # unspecified mask are not supported for sparse COO
                # tensors, see torch.masked.prod implementation
                # for details.
                continue
            sparse_op_kwargs = dict(sample_input_kwargs)
            actual = op(t.to_sparse(), *sample_input.args, **sample_input_kwargs)
            self.assertEqual(actual.layout, torch.sparse_coo)

            expected = op(t, *sample_input.args, **sample_input_kwargs).to_sparse()
            self.assertEqual(actual, expected)


class TestSparseMeta(TestCase):
    exact_dtype = True

    def test_basic(self):
        r = torch.empty(4, 4, layout=torch.sparse_coo, device='meta')
        self.assertTrue(r.is_meta)
        self.assertEqual(r.device.type, "meta")
        r2 = torch.empty_like(r)
        self.assertTrue(r2.is_meta)
        self.assertEqual(r, r2)
        r3 = torch.sparse_coo_tensor(size=(4, 4), device='meta')
        self.assertTrue(r3.is_meta)
        self.assertEqual(r, r3)
        r.sparse_resize_((4, 4), 1, 1)
        r.sparse_resize_and_clear_((4, 4, 4), 2, 1)
        self.assertEqual(r.sparse_dim(), 2)
        self.assertEqual(r.dense_dim(), 1)
        self.assertEqual(r._dimV(), 1)
        self.assertEqual(r._nnz(), 0)
        # nnz zero sparse tensors should always be coalesced at creation
        self.assertEqual(r.is_coalesced(), True)
        # but we can force them into the uncoalesed state
        r._coalesced_(False)
        self.assertEqual(r.is_coalesced(), False)
        # return the coalesced state for indices/values access
        r._coalesced_(True)
        # TODO: this sort of aliasing will need to be handled by
        # functionalization
        self.assertEqual(r._indices(), torch.empty(2, 0, device='meta', dtype=torch.int64))
        self.assertEqual(r._values(), torch.empty(0, 4, device='meta'))
        self.assertEqual(r.indices(), torch.empty(2, 0, device='meta', dtype=torch.int64))
        self.assertEqual(r.values(), torch.empty(0, 4, device='meta'))


class TestSparseAny(TestCase):

    @onlyCPU
    @all_sparse_layouts('layout', include_strided=False)
    @torch.sparse.check_sparse_tensor_invariants(enable=False)
    def test_check_sparse_tensor_invariants(self, layout):

        if layout is torch.sparse_coo:

            def create_invalid_tensor(check_invariants=None):
                shape = (2, 2)
                invalid_indices = torch.tensor([[0], [3]])  # column index is out of range
                values = torch.tensor([1])
                if check_invariants is None:
                    return torch.sparse_coo_tensor(invalid_indices, values, shape)
                else:
                    return torch.sparse_coo_tensor(invalid_indices, values, shape, check_invariants=check_invariants)

            expected_exception_message = 'size is inconsistent with indices: for dim 1, size is 2 but found index 3'

        elif layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:

            def create_invalid_tensor(check_invariants=None):
                shape = (2, 2)
                compressed_indices = torch.tensor([0, 0, 1])
                invalid_plain_indices = torch.tensor([3])  # index is out of range
                if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                    values = torch.tensor([[[1]]])
                else:
                    values = torch.tensor([1])
                if check_invariants is None:
                    return torch.sparse_compressed_tensor(compressed_indices, invalid_plain_indices, values, shape, layout=layout)
                else:
                    return torch.sparse_compressed_tensor(compressed_indices, invalid_plain_indices, values, shape, layout=layout,
                                                          check_invariants=check_invariants)

            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                expected_exception_message = r'`0 <= col_indices < ncols` is not satisfied.'
            else:
                expected_exception_message = r'`0 <= row_indices < nrows` is not satisfied.'

        else:
            raise NotImplementedError(layout)

        # First, consider the case where invariant checks are disabled
        # "globally" (read: within the context of this test method
        # caller) as defined by check_sparse_tensor_invariants(False)
        # decorator:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Enable the invariant checks in a local context:
        with torch.sparse.check_sparse_tensor_invariants():
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Leaving the local context must restore the "global" state of
        # the invariant check feature:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Since invariant checks are disabled by default, we can
        # create an invalid sparse tensor without raising an
        # exception:
        r = create_invalid_tensor()
        self.assertEqual(r.layout, layout)

        # Or, when disabling the invariants check explicitly:
        r = create_invalid_tensor(check_invariants=False)
        self.assertEqual(r.layout, layout)

        # Enabling invariant check via constructor's optional argument
        # will raise an exception when sparse tensor invariants are
        # violated:
        with self.assertRaisesRegex(RuntimeError, expected_exception_message):
            create_invalid_tensor(check_invariants=True)

        # Check that the global invariant check flag has been restored
        # after raising the exception above:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Next, consider the case where invariant checks are enabled
        # within a local context:
        with torch.sparse.check_sparse_tensor_invariants():
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())

            # Since invariant checks are now enabled by default, an
            # attempt to create an invalid sparse tensor will lead to
            # an exception:
            with self.assertRaisesRegex(RuntimeError, expected_exception_message):
                create_invalid_tensor()

            # Similarly, when enabling the invariant checks
            # explicitly, invalid sparse tensor construction will lead
            # to an exception:
            with self.assertRaisesRegex(RuntimeError, expected_exception_message):
                create_invalid_tensor(check_invariants=True)

            # However, invariants check can be disabled via
            # constructor's optional argument so that the invalid
            # tensor is succesfully constructed:
            r = create_invalid_tensor(check_invariants=False)
            self.assertEqual(r.layout, layout)

            # Check that the invariant check flag has been restored
            # when leaving the constructor:
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Double-check restoring the global state when leaving the
        # local context:
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Test nesting of pre-defined context managers
        check_ctx = torch.sparse.check_sparse_tensor_invariants(True)
        no_check_ctx = torch.sparse.check_sparse_tensor_invariants(False)
        with check_ctx:
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            with no_check_ctx:
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

        # Test an attempt to re-use an activate context manager instance
        check_ctx2 = torch.sparse.check_sparse_tensor_invariants(True)
        with check_ctx:
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            with no_check_ctx:
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                with self.assertRaisesRegex(RuntimeError, "This context manager instance is already activated."
                                            " Use a different context manager instance for context nesting"):
                    with check_ctx:
                        self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                with check_ctx2:
                    self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
                self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())
            self.assertTrue(torch.sparse.check_sparse_tensor_invariants.is_enabled())
        self.assertFalse(torch.sparse.check_sparse_tensor_invariants.is_enabled())

    def test_generate_simple_inputs(self):
        layouts = [torch.strided, torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc]

        tested_combinations = set()
        for tensors in zip(*map(self.generate_simple_inputs, layouts)):
            for i, t in enumerate(tensors):
                self.assertEqual(t.layout, layouts[i])

                # all layouts must produce semantically the same tensors
                self.assertEqual(t, tensors[0])

                if t.layout is torch.strided:
                    is_hybrid = None
                else:
                    is_hybrid = t.dense_dim() > 0
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    is_batch = t.crow_indices().ndim > 1
                elif t.layout in {torch.sparse_csc, torch.sparse_bsc}:
                    is_batch = t.ccol_indices().ndim > 1
                else:
                    is_batch = None
                if t.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                    blocksize = t.values().shape[1:3]
                    nontrivial_blocksize = 1 not in blocksize
                else:
                    nontrivial_blocksize = None
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    contiguous_indices = t.crow_indices().is_contiguous() and t.col_indices().is_contiguous()
                    contiguous_values = t.values().is_contiguous()
                elif t.layout in {torch.sparse_csc, torch.sparse_bsc}:
                    contiguous_indices = t.ccol_indices().is_contiguous() and t.row_indices().is_contiguous()
                    contiguous_values = t.values().is_contiguous()
                elif t.layout is torch.sparse_coo:
                    contiguous_indices = t._indices().is_contiguous()
                    contiguous_values = t._values().is_contiguous()
                else:
                    contiguous_indices = None
                    contiguous_values = t.is_contiguous()

                tested_combinations.add((t.layout, is_hybrid, is_batch, nontrivial_blocksize,
                                         contiguous_indices, contiguous_values))

        # Ensure that the inputs generation covers all layout,
        # non-hybrid/hybrid, non-batch/batch, and contiguity
        # combinations:
        untested_combinations = set()
        for layout in layouts:
            for is_hybrid in [False, True]:
                if layout is torch.strided:
                    is_hybrid = None
                for is_batch in [False, True]:
                    if layout in {torch.sparse_coo, torch.strided}:
                        is_batch = None
                    for nontrivial_blocksize in [False, True]:
                        if layout not in {torch.sparse_bsr, torch.sparse_bsc}:
                            nontrivial_blocksize = None
                        for contiguous_indices in [False, True]:
                            if layout is torch.strided:
                                contiguous_indices = None
                            elif not is_batch:
                                # indices are contiguous per-patch
                                contiguous_indices = True
                            for contiguous_values in [False, True]:
                                key = (layout, is_hybrid, is_batch, nontrivial_blocksize,
                                       contiguous_indices, contiguous_values)
                                if key not in tested_combinations:
                                    untested_combinations.add(
                                        f'layout={layout}, is_hybrid={is_hybrid}, is_batch={is_batch},'
                                        f' nontrivial_blocksize={nontrivial_blocksize},'
                                        f' contiguous_indices{contiguous_indices}, contiguous_values={contiguous_values}')
        assert not untested_combinations, untested_combinations

    @all_sparse_layouts('from_layout', include_strided=False)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_to_dense(self, from_layout, device, dtype, index_dtype):
        """
        This test tests conversion from any layout to strided layout.
        """
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype):
            r = t.to_dense()
            self.assertEqual(r.layout, torch.strided)
            self.assertEqual(r, t)

    @all_sparse_layouts('from_layout', include_strided=False)
    @dtypes(torch.float64, torch.complex128)
    @parametrize("index_dtype", [torch.int64])
    @gradcheck_semantics()
    def test_gradcheck_to_dense(self, from_layout, device, dtype, index_dtype, gradcheck):
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype):
            batch_dim = t.dim() - t.dense_dim() - t.sparse_dim()
            if batch_dim > 0:
                # TODO: implement batch support in _convert_indices_from_csr_to_coo
                continue
            t = t.clone().detach().requires_grad_(True)
            r = gradcheck(lambda x: torch.Tensor.to_dense(x, masked_grad=gradcheck.masked), t)
            self.assertTrue(r)

    @all_sparse_layouts('from_layout', include_strided=True)
    @all_sparse_layouts('to_layout', include_strided=False)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_to_sparse(self, from_layout, to_layout, device, dtype, index_dtype):
        """
        This test tests conversion from any layout to any sparse layout.
        """
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype,
                enable_hybrid=(
                    # TODO: to support conversion strided->hybrid
                    # CSR/CSC/BSR/BSC, to_sparse() requires extra keyword
                    # argument, either nof_batch_dims or
                    # nof_dense_dims
                    not (from_layout is torch.strided and to_layout in
                         {torch.sparse_bsr, torch.sparse_bsc, torch.sparse_csr, torch.sparse_csc}))):

            if to_layout in {torch.sparse_bsr, torch.sparse_bsc}:
                if from_layout == torch.sparse_bsr:
                    batch_ndim = t.crow_indices().dim() - 1
                    blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                elif from_layout == torch.sparse_bsc:
                    batch_ndim = t.ccol_indices().dim() - 1
                    blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                else:
                    blocksize = (1, 1)
            else:
                blocksize = None

            if from_layout is torch.strided:
                is_batch = None
                is_hybrid = None
            else:
                is_batch = t.dim() > (t.sparse_dim() + t.dense_dim())
                is_hybrid = t.dense_dim() > 0

            def explicit_to_sparse(x):
                # Used to check that the explicit conversion methods
                # are consistent with the `to_sparse(*, layout,
                # blocksize)` method.
                if to_layout is torch.sparse_coo:
                    return x.to_sparse_coo()
                elif to_layout is torch.sparse_csr:
                    return x.to_sparse_csr()
                elif to_layout is torch.sparse_csc:
                    return x.to_sparse_csc()
                elif to_layout is torch.sparse_bsr:
                    return x.to_sparse_bsr(blocksize)
                elif to_layout is torch.sparse_bsc:
                    return x.to_sparse_bsc(blocksize)
                else:
                    assert 0  # unreachable

            # TODO: The following exception cases all correspond to
            # not implemented conversions
            if from_layout in {
                    torch.sparse_csr, torch.sparse_csc} and to_layout in {torch.sparse_bsr, torch.sparse_bsc} and is_batch:
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse(Csr|Csc) to Sparse(Bsr|Bsc) for batched inputs is not supported"):
                    t.to_sparse(layout=to_layout, blocksize=blocksize)
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse(Csr|Csc) to Sparse(Bsr|Bsc) for batched inputs is not supported"):
                    explicit_to_sparse(t)
                continue
            elif from_layout is torch.sparse_coo and to_layout in {
                    torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc} and t.sparse_dim() != 2:
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse to .* for input tensors with sparse_dim\(\)!=2 is not supported"):
                    t.to_sparse(layout=to_layout, blocksize=blocksize)
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"conversion from Sparse to .* for input tensors with sparse_dim\(\)!=2 is not supported"):
                    explicit_to_sparse(t)
                continue
            elif from_layout in {torch.sparse_csr, torch.sparse_csc,
                                 torch.sparse_bsr, torch.sparse_bsc} and to_layout is torch.sparse_coo and is_batch:
                with self.assertRaisesRegex(RuntimeError,
                                            "crow_indices is supposed to be a vector, but got \\d+ dimensional tensor"):
                    t.to_sparse(layout=to_layout, blocksize=blocksize)
                with self.assertRaisesRegex(RuntimeError,
                                            "crow_indices is supposed to be a vector, but got \\d+ dimensional tensor"):
                    explicit_to_sparse(t)
                continue
            elif (from_layout, to_layout) in {(torch.sparse_bsc, torch.sparse_csr), (torch.sparse_bsc, torch.sparse_csc),
                                              (torch.sparse_bsr, torch.sparse_csr), (torch.sparse_bsr, torch.sparse_csc)}:
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"sparse_compressed_to_sparse_(csr|csc|bsr|bsc): expected\s*(Sparse(Csc|Csr)[,]|)\s*Sparse(Csr|Bsr)"
                        " or Sparse(Csc|Bsc) layout but got Sparse(Csr|Csc|Bsr|Bsc)"):
                    t.to_sparse(layout=to_layout, blocksize=blocksize)
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"sparse_compressed_to_sparse_(csr|csc|bsr|bsc): expected\s*(Sparse(Csc|Csr)[,]|)\s*Sparse(Csr|Bsr)"
                        " or Sparse(Csc|Bsc) layout but got Sparse(Csr|Csc|Bsr|Bsc)"):
                    explicit_to_sparse(t)
                self.skipTest('NOT IMPL')
            else:
                r = t.to_sparse(layout=to_layout, blocksize=blocksize)

                self.assertEqual(r.layout, to_layout)

                # to_sparse method uses unsafe construction of sparse
                # tensors. Here we explicitly validate the results to
                # make sure that the sparse tensors are consistent
                # with the corresponding sparse tensor invariants.
                if r.layout in {torch.sparse_csr, torch.sparse_bsr, torch.sparse_csc, torch.sparse_bsc}:
                    if r.layout in {torch.sparse_csr, torch.sparse_bsr}:
                        compressed_indices, plain_indices = r.crow_indices(), r.col_indices()
                    else:
                        compressed_indices, plain_indices = r.ccol_indices(), r.row_indices()
                    torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, r.values(),
                                                                  r.shape, r.layout)
                    if from_layout in {torch.strided, torch.sparse_coo}:
                        self.assertEqual(compressed_indices.dtype, torch.int64)
                        self.assertEqual(plain_indices.dtype, torch.int64)
                    else:
                        self.assertEqual(compressed_indices.dtype, index_dtype)
                        self.assertEqual(plain_indices.dtype, index_dtype)
                    self.assertEqual(r.values().dtype, dtype)
                elif r.layout is torch.sparse_coo:
                    if t.layout is torch.sparse_coo:
                        self.assertEqual(t.is_coalesced(), r.is_coalesced())

                    # Check r is truly coalesced when r.is_coalesced == True
                    if r.is_coalesced():
                        self.assertTrue(is_coalesced_indices(r))

                    torch._validate_sparse_coo_tensor_args(r._indices(), r._values(), r.shape)
                    self.assertEqual(r._indices().dtype, torch.int64)
                    self.assertEqual(r._values().dtype, dtype)
                else:
                    assert 0  # unreachable

                # Finally, we'll test tensor equality:
                self.assertEqual(r, t)

                # Also, check consistency with explicit conversion methods:
                r2 = explicit_to_sparse(t)
                self.assertEqual(r2, r)

                # Check inverse conversion from sparse compressed block tensors
                if from_layout == torch.sparse_bsr:
                    batch_ndim = t.crow_indices().dim() - 1
                    from_blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                elif from_layout == torch.sparse_bsc:
                    batch_ndim = t.ccol_indices().dim() - 1
                    from_blocksize = t.values().shape[batch_ndim + 1:batch_ndim + 3]
                else:
                    continue
                if r.ndim != 2:
                    continue

                t2 = r.to_sparse(layout=from_layout, blocksize=from_blocksize)
                self.assertEqual(t2, t)

        # extra tests
        if (from_layout, to_layout) == (torch.sparse_csr, torch.sparse_bsr):
            # See gh-90910
            t = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=dtype, device=device).to_sparse_csr()
            r = t.to_sparse_bsr((2, 2))
            torch._validate_sparse_compressed_tensor_args(r.crow_indices(), r.col_indices(), r.values(), r.shape, r.layout)
            self.assertEqual(r, t)

        if (from_layout, to_layout) in {(torch.sparse_csr, torch.sparse_csc),
                                        (torch.sparse_csc, torch.sparse_csr)}:
            # See gh-91007
            compressed_indices = torch.tensor([0, 4, 8, 8, 12, 16, 20], dtype=index_dtype, device=device)
            plain_indices = torch.tensor([0, 1, 2, 3] * 5, dtype=index_dtype, device=device)
            t = torch.sparse_compressed_tensor(compressed_indices, plain_indices, range(20),
                                               dtype=dtype, device=device, layout=from_layout)
            r = t.to_sparse(layout=to_layout)
            if r.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices, plain_indices = r.crow_indices(), r.col_indices()
            else:
                compressed_indices, plain_indices = r.ccol_indices(), r.row_indices()
            torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, r.values(), r.shape, r.layout)
            self.assertEqual(r, t)

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(reduction_ops_with_sparse_support)
    @precisionOverride({torch.bfloat16: 5e-4, torch.float16: 5e-3})
    @all_sparse_layouts('layout', include_strided=False)
    def test_reductions(self, layout, device, dtype, op):
        count = 0
        for sample in op.sample_inputs_sparse(layout, device, dtype):
            count += 1

            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            result = op.op(t_inp, *t_args, **t_kwargs)

            #  Checking invariant rop(inp, ...).to_dense() == rop(inp.to_dense(), ...)
            dense = op.op(t_inp.to_dense(), *t_args, **t_kwargs)
            self.assertEqual(result, dense)

        if count == 0:
            # we count samples to avoid false-positive test reports
            self.skipTest('no sample inputs')

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(reduction_ops_with_sparse_support, allowed_dtypes=(torch.float32, torch.float64, torch.complex64, torch.complex128))
    @all_sparse_layouts('layout', include_strided=False)
    def test_reductions_backward(self, layout, device, dtype, op):
        count = 0
        for sample in op.sample_inputs_sparse(layout, device, dtype, requires_grad=True):
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            r = op.op(t_inp, *t_args, **t_kwargs)
            if r.numel() != 0:
                r = r.sum()

            if op.name == 'sum':
                count += 1
                r.abs().backward()
                self.assertEqual(t_inp.grad, torch.ones(t_inp.shape, dtype=dtype, device=device) * torch.sgn(r))
            else:
                self.skipTest('NOT IMPL')

        if count == 0:
            # we count samples to avoid false-positive test reports
            self.skipTest('no sample inputs')

    @onlyNativeDeviceTypes
    @suppress_warnings
    @parametrize("mth", [subtest(mth, name=mth.__name__)
                         for mth in [torch.Tensor.is_coalesced,
                                     torch.Tensor.coalesce,
                                     torch.Tensor.indices,
                                     torch.Tensor.values,
                                     torch.Tensor.crow_indices,
                                     torch.Tensor.col_indices,
                                     torch.Tensor.ccol_indices,
                                     torch.Tensor.row_indices,
                                     ]])
    @all_sparse_layouts('layout', include_strided=True)
    def test_unsupported_backend_error_message(self, mth, layout, device):
        inp = torch.tensor([[1, 2], [3, 4]], device=device).to_sparse(
            layout=layout,
            blocksize=(1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None)
        assert inp.layout is layout

        expected_behaviour = dict(
            # <mth name> = (<supported layouts>, <exception message on other layouts>)
            is_coalesced=({torch.sparse_coo},
                          "is_coalesced expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            coalesce=({torch.sparse_coo},
                      "coalesce expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            indices=({torch.sparse_coo},
                     "indices expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            values=({torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc},
                    "values expected sparse tensor layout but got Strided"),
            crow_indices=({torch.sparse_csr, torch.sparse_bsr},
                          "crow_indices expected sparse row compressed tensor layout but got (Sparse(Csc|Bsc|)|Strided)"),
            col_indices=({torch.sparse_csr, torch.sparse_bsr},
                         "col_indices expected sparse row compressed tensor layout but got (Sparse(Csc|Bsc|)|Strided)"),
            ccol_indices=({torch.sparse_csc, torch.sparse_bsc},
                          "ccol_indices expected sparse column compressed tensor layout but got (Sparse(Csr|Bsr|)|Strided)"),
            row_indices=({torch.sparse_csc, torch.sparse_bsc},
                         "row_indices expected sparse column compressed tensor layout but got (Sparse(Csr|Bsr|)|Strided)"),
        )[mth.__name__]

        if layout in expected_behaviour[0]:
            mth(inp)
        else:
            with self.assertRaisesRegex(RuntimeError, expected_behaviour[1]):
                mth(inp)

    @onlyNativeDeviceTypes
    @all_sparse_layouts('layout', include_strided=not True)
    @dtypes(torch.float64, torch.cdouble)
    @parametrize("masked", [subtest(False, name='sparse'), subtest(True, name='masked')])
    @parametrize("fast_mode", [subtest(False, name='slow'), subtest(True, name='fast')])
    def test_gradcheck_mm(self, layout, dtype, device, masked, fast_mode):
        # This function does not check the following cases:
        # - batch or hybrid tensors because addmm does not support
        #   such inputs yet
        # - check_forward_ad=True because of the lack of sparse tensor
        #   support in aten::view_as_real, torch._VF._make_dual, etc.

        ref_x = torch.tensor([[1, 2, 0, 0],
                              [0, 6, 0, 0],
                              [0, 0, 0, 0],
                              [13, 14, 0, 15]], dtype=dtype, device=device)
        ref_y = torch.tensor([[11, 12, 13, 14],
                              [21, 22, 23, 24],
                              [31, 32, 33, 34],
                              [41, 42, 43, 44]],
                             dtype=dtype, device=device)

        mm = torch.sparse.mm if masked else torch.mm

        blocksize = (2, 2) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
        x = ref_x.to_sparse(layout=layout, blocksize=blocksize).requires_grad_(True)
        y = ref_y.requires_grad_(True)

        if layout is torch.sparse_bsr and not masked or layout is torch.sparse_bsc:
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"addmm: computation on (CPU|CUDA) is not implemented for Strided \+ Sparse(Bsr|Bsc) @ Strided"):
                torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
            self.skipTest('NOT IMPL')
        elif layout in {torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc} and masked:
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"(sparse_addmm_sparse_backward: unsupported combination of layouts,"
                    r" grad: Strided, mat1: Sparse(Csc|Bsr|Bsc), mat2: Strided"
                    r"|addmm: computation on (CPU|CUDA) is not implemented for "
                    r"Strided \+ Sparse(Csc|Bsr|Bsc) @ Strided without MKL)"):
                torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
            self.skipTest('NOT IMPL')
        else:
            torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)

    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(binary_ufuncs_with_sparse_support)
    @all_sparse_layouts('layout', include_strided=False)
    def test_binary_operation(self, layout, device, dtype, op):
        if not op.supports_sparse_layout(layout):
            self.skipTest(f'{layout} is not supported in `{op.name}` OpInfo definition. Skipping!')

        for sample in op.sample_inputs_sparse(layout, device, dtype):
            if validate_sample_input_sparse(op, sample, check_validate=False) is not sample:
                # that is, the validation returns the sparse sample
                # wrapped within ErrorInput instance
                continue
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
            result = op.op(t_inp, *t_args, **t_kwargs)

            # Check rop(inp, ...).shape == inp.shape
            self.assertEqual(result.shape, t_inp.shape)

            # Check rop(inp, ...).sparse_dim() == inp.sparse_dim()
            self.assertEqual(result.sparse_dim(), t_inp.sparse_dim())

            # Check rop(inp, ...).dense_dim() == inp.dense_dim()
            self.assertEqual(result.dense_dim(), t_inp.dense_dim())

            # Check invariant rop(inp, ...).to_dense() == rop(inp.to_dense(), ...)
            try:
                dense = op.op(t_inp.to_dense(), *(t_args[0].to_dense(), *t_args[1:]), **t_kwargs)
            except Exception as msg:
                # this is strided op issue, so skipping the sample silently here
                if "\"cpublas_axpy_impl\" not implemented for 'ComplexHalf'" in str(msg):
                    continue
                raise
            self.assertEqual(result, dense)


    @onlyCUDA
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @dtypes(torch.int8, torch.half, torch.bfloat16)
    def test_structured_sparse_linear(self, device, dtype):
        def make_tensor(shape, dtype):
            if dtype.is_complex:
                return torch.zeros(shape, dtype=dtype)
            elif dtype.is_floating_point:
                return torch.randn(shape, dtype=dtype) / 10
            else:
                return torch.randint(-5, 5, shape, dtype=dtype)

        def random_mask_choice(i=None):
            choices = [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 1]
            ]
            if i is None:
                i = random.randint(0, len(choices) - 1)
            return choices[i]

        def run_test(batch_shape, m, n, k, device, dtype, dtype_out, add_bias, activation, rtol, atol):
            weight = make_tensor((m, k), dtype).to(device)
            input = make_tensor((*batch_shape, n, k), dtype).to(device)
            bias = make_tensor((m,), dtype_out).to(device) if add_bias else None

            for meta_choice in (list(range(6)) + [None]):
                mask_entries = [random_mask_choice(meta_choice) for i in range(m * (k // 4))]
                mask = torch.tensor(mask_entries, dtype=torch.bool).view(m, k).to(device)
                weight = weight.masked_fill(~mask, 0)

                dtype_dense = torch.float
                input_dense = input.to(dtype_dense)
                weight_dense = weight.to(dtype_dense)
                bias_dense = bias.to(dtype_dense) if add_bias else None
                output0 = torch.nn.functional.linear(input_dense, weight_dense, bias=bias_dense)
                if activation == "relu":
                    relu = torch.nn.ReLU()
                    output0 = relu(output0)
                elif activation == "silu":
                    silu = torch.nn.SiLU()
                    output0 = silu(output0)

                weight_sparse = weight.masked_select(mask).view(m, k // 2)

                output1, meta = torch._structured_sparse_linear(input, weight_sparse, mask, bias=bias, activation=activation)
                torch.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

                output1, _ = torch._structured_sparse_linear(input, weight_sparse, meta, bias=bias, activation=activation)
                torch.testing.assert_close(output1.to(dtype_dense), output0, rtol=rtol, atol=atol)

        is_sm8x = torch.cuda.get_device_capability(0)[0] == 8
        if not is_sm8x:
            return

        batch_shapes = [[], [3], [3, 1]]
        dtype_out = {torch.int8: torch.int32, torch.half: torch.half, torch.bfloat16: torch.bfloat16}
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            rtol, atol = 5e-3, 5e-3
        for (batch_shape, m, n, k, add_bias, activation) in \
                itertools.product(batch_shapes, range(3), range(3), range(3), (False, True), activations):
            if activation == "silu" and dtype == torch.int8:
                continue  # SiLU not supported for integer inputs

            m = 2 ** m * 32
            n = 2 ** n * 32
            k = 2 ** k * 128
            run_test(batch_shape, m, n, k, device, dtype, dtype_out[dtype], add_bias, activation, rtol, atol)

    @onlyCPU
    @all_sparse_layouts('layout', include_strided=True)
    @dtypes(torch.double)
    def test_to_sparse_identity(self, device, layout, dtype):
        for dense_dim in range(4):
            x_dense = torch.eye(dense_dim, dtype=dtype, device=device)
            for sparse_dim_in in range(1, dense_dim):
                x_sparse = x_dense.to_sparse(sparse_dim_in)
                for sparse_dim_out in range(0, dense_dim):
                    if sparse_dim_out == sparse_dim_in:
                        self.assertTrue(x_sparse.to_sparse(sparse_dim_out).sparse_dim() == sparse_dim_out)
                    else:
                        with self.assertRaisesRegex(
                                RuntimeError,
                                r"to_sparse: conversion from Sparse to Sparse with sparse_dim argument !=self.sparse_dim\(\)"
                                " is not supported"):
                            x_sparse.to_sparse(sparse_dim_out)


    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(like_fns_with_sparse_support)
    @all_sparse_layouts('layout', include_strided=False)
    def test_like_fns(self, layout, device, dtype, op):

        for sample in op.sample_inputs_sparse(layout, device, dtype):
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
            if t_inp.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                expected_blocksize = t_inp.values().shape[batch_dim + 1:batch_dim + 3]
            else:
                expected_blocksize = None
            expected_dtype = t_kwargs.get('dtype', dtype)
            expected_device = torch.device(t_kwargs.get('device', device))
            expected_layout = t_kwargs.get('layout', layout)

            result = op.op(t_inp, *t_args, **t_kwargs)

            self.assertEqual(result.dtype, expected_dtype)
            self.assertEqual(result.device.type, expected_device.type)
            self.assertEqual(result.layout, expected_layout)

            if result.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                result_batch_dim = result.dim() - result.dense_dim() - result.sparse_dim()
                blocksize = result.values().shape[result_batch_dim + 1:result_batch_dim + 3]
                self.assertEqual(blocksize, expected_blocksize)

            # Check op(inp).shape == inp.shape
            self.assertEqual(result.shape, t_inp.shape)

            if expected_layout is torch.strided:
                self.assertEqual(result.sparse_dim(), 0)
                # Check op(inp, layout=torch.strided).dense_dim() == inp.dim()
                self.assertEqual(result.dense_dim(), t_inp.dim())
            elif expected_layout is torch.sparse_coo:
                # Check op(inp, layout=torch.sparse_coo).sparse_dim() == batch_dim + inp.sparse_dim()
                self.assertEqual(result.sparse_dim(), batch_dim + t_inp.sparse_dim())
                # Check op(inp, layout=torch.sparse_coo).dense_dim() == inp.dense_dim()
                self.assertEqual(result.dense_dim(), t_inp.dense_dim())

                torch._validate_sparse_coo_tensor_args(result._indices(), result._values(), result.shape)
            else:
                # Check op(inp).sparse_dim() == inp.sparse_dim()
                self.assertEqual(result.sparse_dim(), t_inp.sparse_dim())
                # Check op(inp).dense_dim() == inp.dense_dim()
                self.assertEqual(result.dense_dim(), t_inp.dense_dim())

                if result.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    compressed_indices, plain_indices = result.crow_indices(), result.col_indices()
                else:
                    compressed_indices, plain_indices = result.ccol_indices(), result.row_indices()

                torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, result.values(),
                                                              result.shape, result.layout)

# e.g., TestSparseUnaryUfuncsCPU and TestSparseUnaryUfuncsCUDA
instantiate_device_type_tests(TestSparseUnaryUfuncs, globals(), except_for='meta')

instantiate_device_type_tests(TestSparseMaskedReductions, globals(), except_for='meta')

# e.g., TestSparseCPU and TestSparseCUDA
instantiate_device_type_tests(TestSparse, globals(), except_for='meta')

instantiate_device_type_tests(TestSparseAny, globals(), except_for='meta')

instantiate_parametrized_tests(TestSparseLegacyAndDeprecation)

if __name__ == '__main__':
    run_tests()
