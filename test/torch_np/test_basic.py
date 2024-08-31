# Owner(s): ["module: dynamo"]

import functools
import inspect
from unittest import expectedFailure as xfail, skipIf as skip

import numpy as _np
from pytest import raises as assert_raises

import torch
import torch._numpy as w
import torch._numpy._ufuncs as _ufuncs
import torch._numpy._util as _util
from torch._numpy.testing import assert_allclose, assert_equal
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# These function receive one array_like arg and return one array_like result
one_arg_funcs = [
    w.asarray,
    w.empty_like,
    w.ones_like,
    w.zeros_like,
    functools.partial(w.full_like, fill_value=42),
    w.corrcoef,
    w.squeeze,
    w.argmax,
    # w.bincount,     # XXX: input dtypes
    w.prod,
    w.sum,
    w.real,
    w.imag,
    w.angle,
    w.real_if_close,
    w.isreal,
    w.iscomplex,
    w.isneginf,
    w.isposinf,
    w.i0,
    w.copy,
    w.array,
    w.round,
    w.around,
    w.flip,
    w.vstack,
    w.hstack,
    w.dstack,
    w.column_stack,
    w.row_stack,
    w.flatnonzero,
]

ufunc_names = _ufuncs._unary
ufunc_names.remove("invert")  # torch: bitwise_not_cpu not implemented for 'Float'
ufunc_names.remove("bitwise_not")

one_arg_funcs += [getattr(_ufuncs, name) for name in ufunc_names]


@instantiate_parametrized_tests
class TestOneArr(TestCase):
    """Base for smoke tests of one-arg functions: (array_like) -> (array_like)

    Accepts array_likes, torch.Tensors, w.ndarays; returns an ndarray
    """

    @parametrize("func", one_arg_funcs)
    def test_asarray_tensor(self, func):
        t = torch.Tensor([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t)

        assert isinstance(ta, w.ndarray)

    @parametrize("func", one_arg_funcs)
    def test_asarray_list(self, func):
        lst = [[1.0, 2, 3], [4, 5, 6]]
        la = func(lst)

        assert isinstance(la, w.ndarray)

    @parametrize("func", one_arg_funcs)
    def test_asarray_array(self, func):
        a = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        la = func(a)

        assert isinstance(la, w.ndarray)


one_arg_axis_funcs = [
    w.argmax,
    w.argmin,
    w.prod,
    w.sum,
    w.all,
    w.any,
    w.mean,
    w.argsort,
    w.std,
    w.var,
    w.flip,
]


@instantiate_parametrized_tests
class TestOneArrAndAxis(TestCase):
    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_tensor(self, func, axis):
        t = torch.Tensor([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)

    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_list(self, func, axis):
        t = [[1.0, 2, 3], [4, 5, 6]]
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)

    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_array(self, func, axis):
        t = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        assert isinstance(ta, w.ndarray)


@instantiate_parametrized_tests
class TestOneArrAndAxesTuple(TestCase):
    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_tensor(self, func, axes):
        t = torch.ones((1, 2, 3))
        ta = func(t, axes=axes)
        assert isinstance(ta, w.ndarray)

        # a np.transpose -specific test
        if axes is None:
            newshape = (3, 2, 1)
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(w.ndim(t)))
        assert ta.shape == newshape

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_list(self, func, axes):
        t = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]  # shape = (1, 2, 3)
        ta = func(t, axes=axes)
        assert isinstance(ta, w.ndarray)

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_array(self, func, axes):
        t = w.asarray([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
        ta = func(t, axes=axes)
        assert isinstance(ta, w.ndarray)

        if axes is None:
            newshape = (3, 2, 1)
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(t.ndim))
        assert ta.shape == newshape


arr_shape_funcs = [
    w.reshape,
    w.empty_like,
    w.ones_like,
    functools.partial(w.full_like, fill_value=42),
    w.broadcast_to,
]


@instantiate_parametrized_tests
class TestOneArrAndShape(TestCase):
    """Smoke test of functions (array_like, shape_like) -> array_like"""

    def setUp(self):
        self.shape = (2, 3)
        self.shape_arg_name = {
            w.reshape: "newshape",
        }  # reshape expects `newshape`

    @parametrize("func", arr_shape_funcs)
    def test_andshape_tensor(self, func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        assert isinstance(ta, w.ndarray)
        assert ta.shape == self.shape

    @parametrize("func", arr_shape_funcs)
    def test_andshape_list(self, func):
        t = [[1, 2, 3], [4, 5, 6]]

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        assert isinstance(ta, w.ndarray)
        assert ta.shape == self.shape

    @parametrize("func", arr_shape_funcs)
    def test_andshape_array(self, func):
        t = w.asarray([[1, 2, 3], [4, 5, 6]])

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        assert isinstance(ta, w.ndarray)
        assert ta.shape == self.shape


one_arg_scalar_funcs = [(w.size, _np.size), (w.shape, _np.shape), (w.ndim, _np.ndim)]


@instantiate_parametrized_tests
class TestOneArrToScalar(TestCase):
    """Smoke test of functions (array_like) -> scalar or python object."""

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_tensor(self, func, np_func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(_np.asarray(t))

        assert not isinstance(ta, w.ndarray)
        assert ta == tn

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_list(self, func, np_func):
        t = [[1, 2, 3], [4, 5, 6]]
        ta = func(t)
        tn = np_func(t)

        assert not isinstance(ta, w.ndarray)
        assert ta == tn

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_array(self, func, np_func):
        t = w.asarray([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(t)

        assert not isinstance(ta, w.ndarray)
        assert ta == tn


shape_funcs = [w.zeros, w.empty, w.ones, functools.partial(w.full, fill_value=42)]


@instantiate_parametrized_tests
class TestShapeLikeToArray(TestCase):
    """Smoke test (shape_like) -> array."""

    shape = (3, 4)

    @parametrize("func", shape_funcs)
    def test_shape(self, func):
        a = func(self.shape)

        assert isinstance(a, w.ndarray)
        assert a.shape == self.shape


seq_funcs = [w.atleast_1d, w.atleast_2d, w.atleast_3d, w.broadcast_arrays]


@instantiate_parametrized_tests
class TestSequenceOfArrays(TestCase):
    """Smoke test (sequence of arrays) -> (sequence of arrays)."""

    @parametrize("func", seq_funcs)
    def test_single_tensor(self, func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)

        # for a single argument, broadcast_arrays returns a tuple, while
        # atleast_?d return an array
        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = ta[0] if unpack else ta

        assert isinstance(res, w.ndarray)

    @parametrize("func", seq_funcs)
    def test_single_list(self, func):
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        assert isinstance(res, w.ndarray)

    @parametrize("func", seq_funcs)
    def test_single_array(self, func):
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        assert isinstance(res, w.ndarray)

    @parametrize("func", seq_funcs)
    def test_several(self, func):
        arys = (
            torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            w.asarray([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
        )

        result = func(*arys)
        assert isinstance(result, (tuple, list))
        assert len(result) == len(arys)
        assert all(isinstance(_, w.ndarray) for _ in result)


seq_to_single_funcs = [
    w.concatenate,
    w.stack,
    w.vstack,
    w.hstack,
    w.dstack,
    w.column_stack,
    w.row_stack,
]


@instantiate_parametrized_tests
class TestSequenceOfArraysToSingle(TestCase):
    """Smoke test (sequence of arrays) -> (array)."""

    @parametrize("func", seq_to_single_funcs)
    def test_several(self, func):
        arys = (
            torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            w.asarray([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
        )

        result = func(arys)
        assert isinstance(result, w.ndarray)


single_to_seq_funcs = (
    w.nonzero,
    # https://github.com/Quansight-Labs/numpy_pytorch_interop/pull/121#discussion_r1172824545
    # w.tril_indices_from,
    # w.triu_indices_from,
    w.where,
)


@instantiate_parametrized_tests
class TestArrayToSequence(TestCase):
    """Smoke test array -> (tuple of arrays)."""

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_tensor(self, func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)

        assert isinstance(ta, tuple)
        assert all(isinstance(x, w.ndarray) for x in ta)

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_list(self, func):
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        assert isinstance(la, tuple)
        assert all(isinstance(x, w.ndarray) for x in la)

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_array(self, func):
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        assert isinstance(la, tuple)
        assert all(isinstance(x, w.ndarray) for x in la)


funcs_and_args = [
    (w.linspace, (0, 10, 11)),
    (w.logspace, (1, 2, 5)),
    (w.logspace, (1, 2, 5, 11)),
    (w.geomspace, (1, 1000, 5, 11)),
    (w.eye, (5, 6)),
    (w.identity, (3,)),
    (w.arange, (5,)),
    (w.arange, (5, 8)),
    (w.arange, (5, 8, 0.5)),
    (w.tri, (3, 3, -1)),
]


@instantiate_parametrized_tests
class TestPythonArgsToArray(TestCase):
    """Smoke_test (sequence of scalars) -> (array)"""

    @parametrize("func, args", funcs_and_args)
    def test_argstoarray_simple(self, func, args):
        a = func(*args)
        assert isinstance(a, w.ndarray)


class TestNormalizations(TestCase):
    """Smoke test generic problems with normalizations."""

    def test_unknown_args(self):
        # Check that unknown args to decorated functions fail
        a = w.arange(7) % 2 == 0

        # unknown positional args
        with assert_raises(TypeError):
            w.nonzero(a, "kaboom")

        # unknown kwarg
        with assert_raises(TypeError):
            w.nonzero(a, oops="ouch")

    def test_too_few_args_positional(self):
        with assert_raises(TypeError):
            w.nonzero()

    def test_unknown_args_with_defaults(self):
        # check a function 5 arguments and 4 defaults: this should work
        w.eye(3)

        # five arguments, four defaults: this should fail
        with assert_raises(TypeError):
            w.eye()


class TestCopyTo(TestCase):
    def test_copyto_basic(self):
        dst = w.empty(4)
        src = w.arange(4)
        w.copyto(dst, src)
        assert (dst == src).all()

    def test_copytobcast(self):
        dst = w.empty((4, 2))
        src = w.arange(4)

        # cannot broadcast => error out
        with assert_raises(RuntimeError):
            w.copyto(dst, src)

        # broadcast src against dst
        dst = w.empty((2, 4))
        w.copyto(dst, src)
        assert (dst == src).all()

    def test_copyto_typecast(self):
        dst = w.empty(4, dtype=int)
        src = w.arange(4, dtype=float)

        with assert_raises(TypeError):
            w.copyto(dst, src, casting="no")

        # force the type cast
        w.copyto(dst, src, casting="unsafe")
        assert (dst == src).all()


class TestDivmod(TestCase):
    def test_divmod_out(self):
        x1 = w.arange(8, 15)
        x2 = w.arange(4, 11)

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        assert_equal(quot, x1 // x2)
        assert_equal(rem, x1 % x2)

        out1, out2 = out
        assert quot is out[0]
        assert rem is out[1]

    def test_divmod_out_list(self):
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        assert quot is out[0]
        assert rem is out[1]

    @xfail  # ("out1, out2 not implemented")
    def test_divmod_pos_only(self):
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        out1, out2 = w.empty_like(x1), w.empty_like(x1)

        quot, rem = w.divmod(x1, x2, out1, out2)

        assert quot is out1
        assert rem is out2

    def test_divmod_no_out(self):
        # check that the out= machinery handles no out at all
        x1 = w.array([4, 5, 6])
        x2 = w.array([2, 1, 2])
        quot, rem = w.divmod(x1, x2)

        assert_equal(quot, x1 // x2)
        assert_equal(rem, x1 % x2)

    def test_divmod_out_both_pos_and_kw(self):
        o = w.empty(1)
        with assert_raises(TypeError):
            w.divmod(1, 2, o, o, out=(o, o))


class TestSmokeNotImpl(TestCase):
    def test_nimpl_basic(self):
        # smoke test that the "NotImplemented" annotation is picked up
        with assert_raises(NotImplementedError):
            w.empty(3, like="ooops")


@instantiate_parametrized_tests
class TestDefaultDtype(TestCase):
    def test_defaultdtype_defaults(self):
        # by default, both floats and ints 64 bit
        x = w.empty(3)
        z = x + 1j * x

        assert x.dtype.torch_dtype == torch.float64
        assert z.dtype.torch_dtype == torch.complex128

        assert w.arange(3).dtype.torch_dtype == torch.int64

    @parametrize("dt", ["pytorch", "float32", torch.float32])
    def test_set_default_float(self, dt):
        try:
            w.set_default_dtype(fp_dtype=dt)

            x = w.empty(3)
            z = x + 1j * x

            assert x.dtype.torch_dtype == torch.float32
            assert z.dtype.torch_dtype == torch.complex64

        finally:
            # restore the
            w.set_default_dtype(fp_dtype="numpy")


@skip(_np.__version__ <= "1.23", reason="from_dlpack is new in NumPy 1.23")
class TestExport(TestCase):
    def test_exported_objects(self):
        exported_fns = (
            x
            for x in dir(w)
            if inspect.isfunction(getattr(w, x))
            and not x.startswith("_")
            and x != "set_default_dtype"
        )
        diff = set(exported_fns).difference(set(dir(_np)))
        assert len(diff) == 0, str(diff)


class TestCtorNested(TestCase):
    def test_arrays_in_lists(self):
        lst = [[1, 2], [3, w.array(4)]]
        assert_equal(w.asarray(lst), [[1, 2], [3, 4]])


class TestMisc(TestCase):
    def test_ndarrays_to_tensors(self):
        out = _util.ndarrays_to_tensors(((w.asarray(42), 7), 3))
        assert len(out) == 2
        assert isinstance(out[0], tuple) and len(out[0]) == 2
        assert isinstance(out[0][0], torch.Tensor)

    @skip(not TEST_CUDA, reason="requires cuda")
    def test_f16_on_cuda(self):
        # make sure operations with float16 tensors give same results on CUDA and on CPU
        t = torch.arange(5, dtype=torch.float16)
        assert_allclose(w.vdot(t.cuda(), t.cuda()), w.vdot(t, t))
        assert_allclose(w.inner(t.cuda(), t.cuda()), w.inner(t, t))
        assert_allclose(w.matmul(t.cuda(), t.cuda()), w.matmul(t, t))
        assert_allclose(w.einsum("i,i", t.cuda(), t.cuda()), w.einsum("i,i", t, t))

        assert_allclose(w.mean(t.cuda()), w.mean(t))

        assert_allclose(w.cov(t.cuda(), t.cuda()), w.cov(t, t).tensor.cuda())
        assert_allclose(w.corrcoef(t.cuda()), w.corrcoef(t).tensor.cuda())


if __name__ == "__main__":
    run_tests()
