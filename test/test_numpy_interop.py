# mypy: ignore-errors

# Owner(s): ["module: numpy"]

import sys
from itertools import product

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
    skipMeta,
)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


# For testing handling NumPy objects and sending tensors to / accepting
#   arrays from NumPy.
class TestNumPyInterop(TestCase):
    # Note: the warning this tests for only appears once per program, so
    # other instances of this warning should be addressed to avoid
    # the tests depending on the order in which they're run.
    @onlyCPU
    def test_numpy_non_writeable(self, device):
        arr = np.zeros(5)
        arr.flags["WRITEABLE"] = False
        self.assertWarns(UserWarning, lambda: torch.from_numpy(arr))

    @onlyCPU
    def test_numpy_unresizable(self, device) -> None:
        x = np.zeros((2, 2))
        y = torch.from_numpy(x)  # noqa: F841
        with self.assertRaises(ValueError):
            x.resize((5, 5))

        z = torch.randn(5, 5)
        w = z.numpy()
        with self.assertRaises(RuntimeError):
            z.resize_(10, 10)
        with self.assertRaises(ValueError):
            w.resize((10, 10))

    @onlyCPU
    def test_to_numpy(self, device) -> None:
        def get_castable_tensor(shape, dtype):
            if dtype.is_floating_point:
                dtype_info = torch.finfo(dtype)
                # can't directly use min and max, because for double, max - min
                # is greater than double range and sampling always gives inf.
                low = max(dtype_info.min, -1e10)
                high = min(dtype_info.max, 1e10)
                t = torch.empty(shape, dtype=torch.float64).uniform_(low, high)
            else:
                # can't directly use min and max, because for int64_t, max - min
                # is greater than int64_t range and triggers UB.
                low = max(torch.iinfo(dtype).min, int(-1e10))
                high = min(torch.iinfo(dtype).max, int(1e10))
                t = torch.empty(shape, dtype=torch.int64).random_(low, high)
            return t.to(dtype)

        dtypes = [
            torch.uint8,
            torch.int8,
            torch.short,
            torch.int,
            torch.half,
            torch.float,
            torch.double,
            torch.long,
        ]

        for dtp in dtypes:
            # 1D
            sz = 10
            x = get_castable_tensor(sz, dtp)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            # 1D > 0 storage offset
            xm = get_castable_tensor(sz * 2, dtp)
            x = xm.narrow(0, sz - 1, sz)
            self.assertTrue(x.storage_offset() > 0)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            def check2d(x, y):
                for i in range(sz1):
                    for j in range(sz2):
                        self.assertEqual(x[i][j], y[i][j])

            # empty
            x = torch.tensor([]).to(dtp)
            y = x.numpy()
            self.assertEqual(y.size, 0)

            # contiguous 2D
            sz1 = 3
            sz2 = 5
            x = get_castable_tensor((sz1, sz2), dtp)
            y = x.numpy()
            check2d(x, y)
            self.assertTrue(y.flags["C_CONTIGUOUS"])

            # with storage offset
            xm = get_castable_tensor((sz1 * 2, sz2), dtp)
            x = xm.narrow(0, sz1 - 1, sz1)
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)
            self.assertTrue(y.flags["C_CONTIGUOUS"])

            # non-contiguous 2D
            x = get_castable_tensor((sz2, sz1), dtp).t()
            y = x.numpy()
            check2d(x, y)
            self.assertFalse(y.flags["C_CONTIGUOUS"])

            # with storage offset
            xm = get_castable_tensor((sz2 * 2, sz1), dtp)
            x = xm.narrow(0, sz2 - 1, sz2).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            # non-contiguous 2D with holes
            xm = get_castable_tensor((sz2 * 2, sz1 * 2), dtp)
            x = xm.narrow(0, sz2 - 1, sz2).narrow(1, sz1 - 1, sz1).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            if dtp != torch.half:
                # check writeable
                x = get_castable_tensor((3, 4), dtp)
                y = x.numpy()
                self.assertTrue(y.flags.writeable)
                y[0][1] = 3
                self.assertTrue(x[0][1] == 3)
                y = x.t().numpy()
                self.assertTrue(y.flags.writeable)
                y[0][1] = 3
                self.assertTrue(x[0][1] == 3)

    def test_to_numpy_bool(self, device) -> None:
        x = torch.tensor([True, False], dtype=torch.bool)
        self.assertEqual(x.dtype, torch.bool)

        y = x.numpy()
        self.assertEqual(y.dtype, np.bool_)
        for i in range(len(x)):
            self.assertEqual(x[i], y[i])

        x = torch.tensor([True], dtype=torch.bool)
        self.assertEqual(x.dtype, torch.bool)

        y = x.numpy()
        self.assertEqual(y.dtype, np.bool_)
        self.assertEqual(x[0], y[0])

    @skipIfTorchDynamo("conj bit not implemented in TensorVariable yet")
    def test_to_numpy_force_argument(self, device) -> None:
        for force in [False, True]:
            for requires_grad in [False, True]:
                for sparse in [False, True]:
                    for conj in [False, True]:
                        data = [[1 + 2j, -2 + 3j], [-1 - 2j, 3 - 2j]]
                        x = torch.tensor(
                            data, requires_grad=requires_grad, device=device
                        )
                        y = x
                        if sparse:
                            if requires_grad:
                                continue
                            x = x.to_sparse()
                        if conj:
                            x = x.conj()
                            y = x.resolve_conj()
                        expect_error = (
                            requires_grad or sparse or conj or not device == "cpu"
                        )
                        error_msg = r"Use (t|T)ensor\..*(\.numpy\(\))?"
                        if not force and expect_error:
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError), error_msg, lambda: x.numpy()
                            )
                            self.assertRaisesRegex(
                                (RuntimeError, TypeError),
                                error_msg,
                                lambda: x.numpy(force=False),
                            )
                        elif force and sparse:
                            self.assertRaisesRegex(
                                TypeError, error_msg, lambda: x.numpy(force=True)
                            )
                        else:
                            self.assertEqual(x.numpy(force=force), y)

    def test_from_numpy(self, device) -> None:
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.complex64,
            np.complex128,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            np.longlong,
            np.bool_,
        ]
        complex_dtypes = [
            np.complex64,
            np.complex128,
        ]

        for dtype in dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)
            tensor_from_array = torch.from_numpy(array)
            # TODO: change to tensor equality check once HalfTensor
            # implements `==`
            for i in range(len(array)):
                self.assertEqual(tensor_from_array[i], array[i])
            # ufunc 'remainder' not supported for complex dtypes
            if dtype not in complex_dtypes:
                # This is a special test case for Windows
                # https://github.com/pytorch/pytorch/issues/22615
                array2 = array % 2
                tensor_from_array2 = torch.from_numpy(array2)
                for i in range(len(array2)):
                    self.assertEqual(tensor_from_array2[i], array2[i])

        # Test unsupported type
        array = np.array(["foo", "bar"], dtype=np.dtype(np.str_))
        with self.assertRaises(TypeError):
            tensor_from_array = torch.from_numpy(array)

        # check storage offset
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[1]
        expected = torch.arange(1, 126, dtype=torch.float64).view(5, 5, 5)[1]
        self.assertEqual(torch.from_numpy(x), expected)

        # check noncontiguous
        x = np.linspace(1, 25, 25)
        x.shape = (5, 5)
        expected = torch.arange(1, 26, dtype=torch.float64).view(5, 5).t()
        self.assertEqual(torch.from_numpy(x.T), expected)

        # check noncontiguous with holes
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[:, 1]
        expected = torch.arange(1, 126, dtype=torch.float64).view(5, 5, 5)[:, 1]
        self.assertEqual(torch.from_numpy(x), expected)

        # check zero dimensional
        x = np.zeros((0, 2))
        self.assertEqual(torch.from_numpy(x).shape, (0, 2))
        x = np.zeros((2, 0))
        self.assertEqual(torch.from_numpy(x).shape, (2, 0))

        # check ill-sized strides raise exception
        x = np.array([3.0, 5.0, 8.0])
        x.strides = (3,)
        self.assertRaises(ValueError, lambda: torch.from_numpy(x))

    @skipIfTorchDynamo("No need to test invalid dtypes that should fail by design.")
    def test_from_numpy_no_leak_on_invalid_dtype(self):
        # This used to leak memory as the `from_numpy` call raised an exception and didn't decref the temporary
        # object. See https://github.com/pytorch/pytorch/issues/121138
        x = np.array("value".encode("ascii"))
        for _ in range(1000):
            try:
                torch.from_numpy(x)
            except TypeError:
                pass
        self.assertTrue(sys.getrefcount(x) == 2)

    @skipMeta
    def test_from_list_of_ndarray_warning(self, device):
        warning_msg = (
            r"Creating a tensor from a list of numpy.ndarrays is extremely slow"
        )
        with self.assertWarnsOnceRegex(UserWarning, warning_msg):
            torch.tensor([np.array([0]), np.array([1])], device=device)

    def test_ctor_with_invalid_numpy_array_sequence(self, device):
        # Invalid list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor(
                [np.random.random(size=(3, 3)), np.random.random(size=(3, 0))],
                device=device,
            )

        # Invalid list of list of numpy array
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor(
                [[np.random.random(size=(3, 3)), np.random.random(size=(3, 2))]],
                device=device,
            )

        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor(
                [
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 3))],
                    [np.random.random(size=(3, 3)), np.random.random(size=(3, 2))],
                ],
                device=device,
            )

        # expected shape is `[1, 2, 3]`, hence we try to iterate over 0-D array
        # leading to type error : not a sequence.
        with self.assertRaisesRegex(TypeError, "not a sequence"):
            torch.tensor(
                [[np.random.random(size=(3)), np.random.random()]], device=device
            )

        # list of list or numpy array.
        with self.assertRaisesRegex(ValueError, "expected sequence of length"):
            torch.tensor([[1, 2, 3], np.random.random(size=(2,))], device=device)

    @onlyCPU
    def test_ctor_with_numpy_scalar_ctor(self, device) -> None:
        dtypes = [
            np.double,
            np.float64,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.uint8,
            np.bool_,
        ]
        for dtype in dtypes:
            self.assertEqual(dtype(42), torch.tensor(dtype(42)).item())

    @onlyCPU
    def test_numpy_index(self, device):
        i = np.array([0, 1, 2], dtype=np.int32)
        x = torch.randn(5, 5)
        for idx in i:
            self.assertFalse(isinstance(idx, int))
            self.assertEqual(x[idx], x[int(idx)])

    @onlyCPU
    def test_numpy_index_multi(self, device):
        for dim_sz in [2, 8, 16, 32]:
            i = np.zeros((dim_sz, dim_sz, dim_sz), dtype=np.int32)
            i[: dim_sz // 2, :, :] = 1
            x = torch.randn(dim_sz, dim_sz, dim_sz)
            self.assertTrue(x[i == 1].numel() == np.sum(i))

    @onlyCPU
    def test_numpy_array_interface(self, device):
        types = [
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.HalfTensor,
            torch.LongTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.ByteTensor,
        ]
        dtypes = [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.uint8,
        ]
        for tp, dtype in zip(types, dtypes):
            # Only concrete class can be given where "Type[number[_64Bit]]" is expected
            if np.dtype(dtype).kind == "u":  # type: ignore[misc]
                # .type expects a XxxTensor, which have no type hints on
                # purpose, so ignore during mypy type checking
                x = torch.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
                array = np.array([1, 2, 3, 4], dtype=dtype)
            else:
                x = torch.tensor([1, -2, 3, -4]).type(tp)  # type: ignore[call-overload]
                array = np.array([1, -2, 3, -4], dtype=dtype)

            # Test __array__ w/o dtype argument
            asarray = np.asarray(x)
            self.assertIsInstance(asarray, np.ndarray)
            self.assertEqual(asarray.dtype, dtype)
            for i in range(len(x)):
                self.assertEqual(asarray[i], x[i])

            # Test __array_wrap__, same dtype
            abs_x = np.abs(x)
            abs_array = np.abs(array)
            self.assertIsInstance(abs_x, tp)
            for i in range(len(x)):
                self.assertEqual(abs_x[i], abs_array[i])

        # Test __array__ with dtype argument
        for dtype in dtypes:
            x = torch.IntTensor([1, -2, 3, -4])
            asarray = np.asarray(x, dtype=dtype)
            self.assertEqual(asarray.dtype, dtype)
            # Only concrete class can be given where "Type[number[_64Bit]]" is expected
            if np.dtype(dtype).kind == "u":  # type: ignore[misc]
                wrapped_x = np.array([1, -2, 3, -4]).astype(dtype)
                for i in range(len(x)):
                    self.assertEqual(asarray[i], wrapped_x[i])
            else:
                for i in range(len(x)):
                    self.assertEqual(asarray[i], x[i])

        # Test some math functions with float types
        float_types = [torch.DoubleTensor, torch.FloatTensor]
        float_dtypes = [np.float64, np.float32]
        for tp, dtype in zip(float_types, float_dtypes):
            x = torch.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
            array = np.array([1, 2, 3, 4], dtype=dtype)
            for func in ["sin", "sqrt", "ceil"]:
                ufunc = getattr(np, func)
                res_x = ufunc(x)
                res_array = ufunc(array)
                self.assertIsInstance(res_x, tp)
                for i in range(len(x)):
                    self.assertEqual(res_x[i], res_array[i])

        # Test functions with boolean return value
        for tp, dtype in zip(types, dtypes):
            x = torch.tensor([1, 2, 3, 4]).type(tp)  # type: ignore[call-overload]
            array = np.array([1, 2, 3, 4], dtype=dtype)
            geq2_x = np.greater_equal(x, 2)
            geq2_array = np.greater_equal(array, 2).astype("uint8")
            self.assertIsInstance(geq2_x, torch.ByteTensor)
            for i in range(len(x)):
                self.assertEqual(geq2_x[i], geq2_array[i])

    @onlyCPU
    def test_multiplication_numpy_scalar(self, device) -> None:
        for np_dtype in [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.int16,
            np.uint8,
        ]:
            for t_dtype in [torch.float, torch.double]:
                # mypy raises an error when np.floatXY(2.0) is called
                # even though this is valid code
                np_sc = np_dtype(2.0)  # type: ignore[abstract, arg-type]
                t = torch.ones(2, requires_grad=True, dtype=t_dtype)
                r1 = t * np_sc
                self.assertIsInstance(r1, torch.Tensor)
                self.assertTrue(r1.dtype == t_dtype)
                self.assertTrue(r1.requires_grad)
                r2 = np_sc * t
                self.assertIsInstance(r2, torch.Tensor)
                self.assertTrue(r2.dtype == t_dtype)
                self.assertTrue(r2.requires_grad)

    @onlyCPU
    @skipIfTorchDynamo()
    def test_parse_numpy_int_overflow(self, device):
        # assertRaises uses a try-except which dynamo has issues with
        # Only concrete class can be given where "Type[number[_64Bit]]" is expected
        if np.__version__ > "2":
            self.assertRaisesRegex(
                OverflowError,
                "out of bounds",
                lambda: torch.mean(torch.randn(1, 1), np.uint64(-1)),
            )  # type: ignore[call-overload]
        else:
            self.assertRaisesRegex(
                RuntimeError,
                "(Overflow|an integer is required)",
                lambda: torch.mean(torch.randn(1, 1), np.uint64(-1)),
            )  # type: ignore[call-overload]

    @onlyCPU
    def test_parse_numpy_int(self, device):
        # https://github.com/pytorch/pytorch/issues/29252
        for nptype in [np.int16, np.int8, np.uint8, np.int32, np.int64]:
            scalar = 3
            np_arr = np.array([scalar], dtype=nptype)
            np_val = np_arr[0]

            # np integral type can be treated as a python int in native functions with
            # int parameters:
            self.assertEqual(torch.ones(5).diag(scalar), torch.ones(5).diag(np_val))
            self.assertEqual(
                torch.ones([2, 2, 2, 2]).mean(scalar),
                torch.ones([2, 2, 2, 2]).mean(np_val),
            )

            # numpy integral type parses like a python int in custom python bindings:
            self.assertEqual(torch.Storage(np_val).size(), scalar)  # type: ignore[attr-defined]

            tensor = torch.tensor([2], dtype=torch.int)
            tensor[0] = np_val
            self.assertEqual(tensor[0], np_val)

            # Original reported issue, np integral type parses to the correct
            # PyTorch integral type when passed for a `Scalar` parameter in
            # arithmetic operations:
            t = torch.from_numpy(np_arr)
            self.assertEqual((t + np_val).dtype, t.dtype)
            self.assertEqual((np_val + t).dtype, t.dtype)

    def test_has_storage_numpy(self, device):
        for dtype in [np.float32, np.float64, np.int64, np.int32, np.int16, np.uint8]:
            arr = np.array([1], dtype=dtype)
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.float32).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.double).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.int).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.long).storage()
            )
            self.assertIsNotNone(
                torch.tensor(arr, device=device, dtype=torch.uint8).storage()
            )

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_numpy_scalar_cmp(self, device, dtype):
        if dtype.is_complex:
            tensors = (
                torch.tensor(complex(1, 3), dtype=dtype, device=device),
                torch.tensor([complex(1, 3), 0, 2j], dtype=dtype, device=device),
                torch.tensor(
                    [[complex(3, 1), 0], [-1j, 5]], dtype=dtype, device=device
                ),
            )
        else:
            tensors = (
                torch.tensor(3, dtype=dtype, device=device),
                torch.tensor([1, 0, -3], dtype=dtype, device=device),
                torch.tensor([[3, 0, -1], [3, 5, 4]], dtype=dtype, device=device),
            )

        for tensor in tensors:
            if dtype == torch.bfloat16:
                with self.assertRaises(TypeError):
                    np_array = tensor.cpu().numpy()
                continue

            np_array = tensor.cpu().numpy()
            for t, a in product(
                (tensor.flatten()[0], tensor.flatten()[0].item()),
                (np_array.flatten()[0], np_array.flatten()[0].item()),
            ):
                self.assertEqual(t, a)
                if (
                    dtype == torch.complex64
                    and torch.is_tensor(t)
                    and type(a) == np.complex64
                ):
                    # TODO: Imaginary part is dropped in this case. Need fix.
                    # https://github.com/pytorch/pytorch/issues/43579
                    self.assertFalse(t == a)
                else:
                    self.assertTrue(t == a)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    def test___eq__(self, device, dtype):
        a = make_tensor((5, 7), dtype=dtype, device=device, low=-9, high=9)
        b = a.detach().clone()
        b_np = b.numpy()

        # Check all elements equal
        res_check = torch.ones_like(a, dtype=torch.bool)
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check one element unequal
        if dtype == torch.bool:
            b[1][3] = not b[1][3]
        else:
            b[1][3] += 1
        res_check[1][3] = False
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check random elements unequal
        rand = torch.randint(0, 2, a.shape, dtype=torch.bool)
        res_check = rand.logical_not()
        b.copy_(a)

        if dtype == torch.bool:
            b[rand] = b[rand].logical_not()
        else:
            b[rand] += 1

        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

        # Check all elements unequal
        if dtype == torch.bool:
            b.copy_(a.logical_not())
        else:
            b.copy_(a + 1)
        res_check.fill_(False)
        self.assertEqual(a == b_np, res_check)
        self.assertEqual(b_np == a, res_check)

    @onlyCPU
    def test_empty_tensors_interop(self, device):
        x = torch.rand((), dtype=torch.float16)
        y = torch.tensor(np.random.rand(0), dtype=torch.float16)
        # Same can be achieved by running
        # y = torch.empty_strided((0,), (0,), dtype=torch.float16)

        # Regression test for https://github.com/pytorch/pytorch/issues/115068
        self.assertEqual(torch.true_divide(x, y).shape, y.shape)
        # Regression test for https://github.com/pytorch/pytorch/issues/115066
        self.assertEqual(torch.mul(x, y).shape, y.shape)
        # Regression test for https://github.com/pytorch/pytorch/issues/113037
        self.assertEqual(torch.div(x, y, rounding_mode="floor").shape, y.shape)


instantiate_device_type_tests(TestNumPyInterop, globals())

if __name__ == "__main__":
    run_tests()
