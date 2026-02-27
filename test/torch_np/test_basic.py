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

        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", one_arg_funcs)
    def test_asarray_list(self, func):
        lst = [[1.0, 2, 3], [4, 5, 6]]
        la = func(lst)

        if not isinstance(la, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(la)}")

    @parametrize("func", one_arg_funcs)
    def test_asarray_array(self, func):
        a = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        la = func(a)

        if not isinstance(la, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(la)}")


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
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_list(self, func, axis):
        t = [[1.0, 2, 3], [4, 5, 6]]
        ta = func(t, axis=axis)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", one_arg_axis_funcs)
    @parametrize("axis", [0, 1, -1, None])
    def test_andaxis_array(self, func, axis):
        t = w.asarray([[1.0, 2, 3], [4, 5, 6]])
        ta = func(t, axis=axis)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")


@instantiate_parametrized_tests
class TestOneArrAndAxesTuple(TestCase):
    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_tensor(self, func, axes):
        t = torch.ones((1, 2, 3))
        ta = func(t, axes=axes)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

        # a np.transpose -specific test
        if axes is None:
            newshape = (3, 2, 1)
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(w.ndim(t)))
        if ta.shape != newshape:
            raise AssertionError(f"Expected shape {newshape}, got {ta.shape}")

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_list(self, func, axes):
        t = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]  # shape = (1, 2, 3)
        ta = func(t, axes=axes)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

    @parametrize("func", [w.transpose])
    @parametrize("axes", [(0, 2, 1), (1, 2, 0), None])
    def test_andtuple_array(self, func, axes):
        t = w.asarray([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])
        ta = func(t, axes=axes)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")

        if axes is None:
            newshape = (3, 2, 1)
        else:
            newshape = tuple(t.shape[axes[i]] for i in range(t.ndim))
        if ta.shape != newshape:
            raise AssertionError(f"Expected shape {newshape}, got {ta.shape}")


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
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")
        if ta.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {ta.shape}")

    @parametrize("func", arr_shape_funcs)
    def test_andshape_list(self, func):
        t = [[1, 2, 3], [4, 5, 6]]

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")
        if ta.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {ta.shape}")

    @parametrize("func", arr_shape_funcs)
    def test_andshape_array(self, func):
        t = w.asarray([[1, 2, 3], [4, 5, 6]])

        shape_dict = {self.shape_arg_name.get(func, "shape"): self.shape}
        ta = func(t, **shape_dict)
        if not isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(ta)}")
        if ta.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {ta.shape}")


one_arg_scalar_funcs = [(w.size, _np.size), (w.shape, _np.shape), (w.ndim, _np.ndim)]


@instantiate_parametrized_tests
class TestOneArrToScalar(TestCase):
    """Smoke test of functions (array_like) -> scalar or python object."""

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_tensor(self, func, np_func):
        t = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(_np.asarray(t))

        if isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected scalar result, got {type(ta)}")
        if ta != tn:
            raise AssertionError(f"Expected {tn}, got {ta}")

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_list(self, func, np_func):
        t = [[1, 2, 3], [4, 5, 6]]
        ta = func(t)
        tn = np_func(t)

        if isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected scalar result, got {type(ta)}")
        if ta != tn:
            raise AssertionError(f"Expected {tn}, got {ta}")

    @parametrize("func, np_func", one_arg_scalar_funcs)
    def test_toscalar_array(self, func, np_func):
        t = w.asarray([[1, 2, 3], [4, 5, 6]])
        ta = func(t)
        tn = np_func(t)

        if isinstance(ta, w.ndarray):
            raise AssertionError(f"Expected scalar result, got {type(ta)}")
        if ta != tn:
            raise AssertionError(f"Expected {tn}, got {ta}")


shape_funcs = [w.zeros, w.empty, w.ones, functools.partial(w.full, fill_value=42)]


@instantiate_parametrized_tests
class TestShapeLikeToArray(TestCase):
    """Smoke test (shape_like) -> array."""

    shape = (3, 4)

    @parametrize("func", shape_funcs)
    def test_shape(self, func):
        a = func(self.shape)

        if not isinstance(a, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(a)}")
        if a.shape != self.shape:
            raise AssertionError(f"Expected shape {self.shape}, got {a.shape}")


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

        if not isinstance(res, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(res)}")

    @parametrize("func", seq_funcs)
    def test_single_list(self, func):
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        if not isinstance(res, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(res)}")

    @parametrize("func", seq_funcs)
    def test_single_array(self, func):
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        unpack = {w.broadcast_arrays: True}.get(func, False)
        res = la[0] if unpack else la

        if not isinstance(res, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(res)}")

    @parametrize("func", seq_funcs)
    def test_several(self, func):
        arys = (
            torch.Tensor([[1, 2, 3], [4, 5, 6]]),
            w.asarray([[1, 2, 3], [4, 5, 6]]),
            [[1, 2, 3], [4, 5, 6]],
        )

        result = func(*arys)
        if not isinstance(result, (tuple, list)):
            raise AssertionError(f"Expected tuple or list, got {type(result)}")
        if len(result) != len(arys):
            raise AssertionError(f"Expected {len(arys)} results, got {len(result)}")
        if not all(isinstance(_, w.ndarray) for _ in result):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in result]}"
            )


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
        if not isinstance(result, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(result)}")


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

        if not isinstance(ta, tuple):
            raise AssertionError(f"Expected tuple, got {type(ta)}")
        if not all(isinstance(x, w.ndarray) for x in ta):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in ta]}"
            )

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_list(self, func):
        lst = [[1, 2, 3], [4, 5, 6]]
        la = func(lst)

        if not isinstance(la, tuple):
            raise AssertionError(f"Expected tuple, got {type(la)}")
        if not all(isinstance(x, w.ndarray) for x in la):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in la]}"
            )

    @parametrize("func", single_to_seq_funcs)
    def test_asarray_array(self, func):
        a = w.asarray([[1, 2, 3], [4, 5, 6]])
        la = func(a)

        if not isinstance(la, tuple):
            raise AssertionError(f"Expected tuple, got {type(la)}")
        if not all(isinstance(x, w.ndarray) for x in la):
            raise AssertionError(
                f"Expected all items to be w.ndarray, got {[type(x) for x in la]}"
            )


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
        if not isinstance(a, w.ndarray):
            raise AssertionError(f"Expected w.ndarray, got {type(a)}")


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
        if not (dst == src).all():
            raise AssertionError("Expected dst to match src after copyto")

    def test_copytobcast(self):
        dst = w.empty((4, 2))
        src = w.arange(4)

        # cannot broadcast => error out
        with assert_raises(RuntimeError):
            w.copyto(dst, src)

        # broadcast src against dst
        dst = w.empty((2, 4))
        w.copyto(dst, src)
        if not (dst == src).all():
            raise AssertionError("Expected dst to match src after broadcast copyto")

    def test_copyto_typecast(self):
        dst = w.empty(4, dtype=int)
        src = w.arange(4, dtype=float)

        with assert_raises(TypeError):
            w.copyto(dst, src, casting="no")

        # force the type cast
        w.copyto(dst, src, casting="unsafe")
        if not (dst == src).all():
            raise AssertionError("Expected dst to match src after type cast copyto")


class TestDivmod(TestCase):
    def test_divmod_out(self):
        x1 = w.arange(8, 15)
        x2 = w.arange(4, 11)

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        assert_equal(quot, x1 // x2)
        assert_equal(rem, x1 % x2)

        out1, out2 = out
        if quot is not out1:
            raise AssertionError("Expected quot to be out1")
        if rem is not out2:
            raise AssertionError("Expected rem to be out2")

    def test_divmod_out_list(self):
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        out = (w.empty_like(x1), w.empty_like(x1))

        quot, rem = w.divmod(x1, x2, out=out)

        if quot is not out[0]:
            raise AssertionError("Expected quot to be out[0]")
        if rem is not out[1]:
            raise AssertionError("Expected rem to be out[1]")

    @xfail  # ("out1, out2 not implemented")
    def test_divmod_pos_only(self):
        x1 = [4, 5, 6]
        x2 = [2, 1, 2]

        out1, out2 = w.empty_like(x1), w.empty_like(x1)

        quot, rem = w.divmod(x1, x2, out1, out2)

        if quot is not out1:
            raise AssertionError("Expected quot to be out1")
        if rem is not out2:
            raise AssertionError("Expected rem to be out2")

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

        if x.dtype.torch_dtype != torch.float64:
            raise AssertionError(f"Expected torch.float64, got {x.dtype.torch_dtype}")
        if z.dtype.torch_dtype != torch.complex128:
            raise AssertionError(
                f"Expected torch.complex128, got {z.dtype.torch_dtype}"
            )

        if w.arange(3).dtype.torch_dtype != torch.int64:
            raise AssertionError(
                f"Expected torch.int64, got {w.arange(3).dtype.torch_dtype}"
            )

    @parametrize("dt", ["pytorch", "float32", torch.float32])
    def test_set_default_float(self, dt):
        try:
            w.set_default_dtype(fp_dtype=dt)

            x = w.empty(3)
            z = x + 1j * x

            if x.dtype.torch_dtype != torch.float32:
                raise AssertionError(
                    f"Expected torch.float32, got {x.dtype.torch_dtype}"
                )
            if z.dtype.torch_dtype != torch.complex64:
                raise AssertionError(
                    f"Expected torch.complex64, got {z.dtype.torch_dtype}"
                )

        finally:
            # restore the
            w.set_default_dtype(fp_dtype="numpy")


@skip(_np.__version__ <= "1.23", reason="from_dlpack is new in NumPy 1.23")
class TestExport(TestCase):
    def test_exported_objects(self):
        exported_fns = {
            x
            for x in dir(w)
            if inspect.isfunction(getattr(w, x))
            and not x.startswith("_")
            and x != "set_default_dtype"
        }
        if _np.__version__ > "2":
            # The following methods are removed in NumPy 2.
            # See https://numpy.org/devdocs/numpy_2_0_migration_guide.html#main-namespace
            exported_fns -= {"product", "round_", "sometrue", "cumproduct", "alltrue"}

        diff = exported_fns.difference(set(dir(_np)))
        if len(diff) != 0:
            raise AssertionError(str(diff))


class TestCtorNested(TestCase):
    def test_arrays_in_lists(self):
        lst = [[1, 2], [3, w.array(4)]]
        assert_equal(w.asarray(lst), [[1, 2], [3, 4]])


class TestMisc(TestCase):
    def test_ndarrays_to_tensors(self):
        out = _util.ndarrays_to_tensors(((w.asarray(42), 7), 3))
        if len(out) != 2:
            raise AssertionError(f"Expected 2 outputs, got {len(out)}")
        if not isinstance(out[0], tuple) or len(out[0]) != 2:
            raise AssertionError(
                f"Expected out[0] to be tuple of len 2, got {type(out[0])} len {len(out[0])}"
            )
        if not isinstance(out[0][0], torch.Tensor):
            raise AssertionError(
                f"Expected out[0][0] to be torch.Tensor, got {type(out[0][0])}"
            )

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
