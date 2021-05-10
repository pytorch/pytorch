import torch
import numpy as np

import unittest
from itertools import product, permutations, combinations
from functools import partial
import random

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, suppress_warnings, make_tensor)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCPU, dtypes, onlyOnCPUAndCUDA)

# TODO: replace this with make_tensor() in common_utils.py
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x

# TODO: replace this with make_tensor() in common_utils.py
def _rand_shape(dim, min_size, max_size):
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

# TODO: refactor tests to avoid this function
# Converts half/bfloat16 dtype to float when device is cpu
def _convert_t(dtype, device):
    if device == 'cpu' and dtype in {torch.half, torch.bfloat16}:
        return torch.float
    return dtype

# TODO: replace this with make_tensor() in common_utils.py
# Returns a tensor of the requested shape, dtype, and device
# Requesting a half CPU tensor returns a float CPU tensor with
# values representable by a half.
# Initialization uses randint for non-float types and randn for float types.
def _make_tensor(shape, dtype, device, fill_ones=False) -> torch.Tensor:
    # Returns a tensor filled with ones
    if fill_ones:
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    # Returns a tensor with random integer values
    if not (dtype.is_floating_point or dtype.is_complex):
        t = torch.randint(0, 10, shape, device=device)
        if dtype != torch.uint8:
            t = t - 5  # generate negative values also
        return t.to(_convert_t(dtype, device))

    # Populates the CPU tensor with floats representable as half/bfloat16
    if dtype == torch.half and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).half().float()
    if dtype == torch.bfloat16 and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16().float()

    # Default: returns a tensor with random float values
    return torch.randn(shape, dtype=dtype, device=device).to(dtype=dtype)

# Tests ops and indexing to ensure they return views (and new tensors) as
# appropriate.
class TestViewOps(TestCase):
    exact_dtype = True

    def is_view_of(self, base, other):
        if (not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device):
            return False
        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        if base.device.type == 'cpu' or base.device.type == 'cuda':
            if base.storage().data_ptr() != other.storage().data_ptr():
                return False

        return True

    # Returns true if v1 and v2 are views of the same base
    def is_view_of_same_base(self, v1, v2):
        if (not v1._is_view() or v1 is v2):
            return False
        return self.is_view_of(v1._base, v2)

    # Performs transpose if contiguous=True, else returns the input tensor as is
    def _do_transpose(self, x, contiguous=False, dim0=0, dim1=1):
        if contiguous:
            return x
        else:
            return x.transpose(dim0, dim1)

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_conj_self(self, device, dtype):
        t = torch.ones(5, 5, device=device)
        s = t.conj()
        self.assertTrue(s is t)

    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_fp_dtypes(include_bfloat16=False), torch.complex64)
    def test_view_dtype(self, device, dtype):
        int_dtype = {
            torch.half: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float: torch.int,
            torch.double: torch.long,
            torch.complex64: torch.long,
        }[dtype]
        numpy_dtype = {
            torch.half: np.int16,
            torch.bfloat16: np.int16,
            torch.float: np.int32,
            torch.double: np.int64,
            torch.complex64: np.int64,
        }[dtype]

        def generate_inputs():
            yield make_tensor((5, 5, 5), device, dtype, low=-5, high=5)
            yield make_tensor((5, 5, 5), device, dtype, low=-5, high=5).permute(2, 0, 1)
            yield make_tensor((1, 5, 1), device, dtype, low=-5, high=5).expand(5, 5, 5)
            yield make_tensor((10, 5, 10), device, dtype, low=-5, high=5)[::2, :, ::2]
            yield make_tensor((0, 5, 10), device, dtype, low=-5, high=5)
            yield make_tensor((), device, dtype, low=-5, high=5)

        def run_test(fp_tensor):
            self.assertRaises(RuntimeError, lambda: fp_tensor.view(torch.complex128))
            self.assertRaises(RuntimeError, lambda: fp_tensor.view(torch.int8))

            int_tensor = fp_tensor.view(int_dtype)
            self.assertEqual(int_tensor.dtype, int_dtype)
            self.assertEqual(int_tensor.shape, fp_tensor.shape)
            self.assertEqual(int_tensor.stride(), fp_tensor.stride())

            self.assertEqual(fp_tensor, int_tensor.view(dtype), rtol=0, atol=0)
            self.assertEqual(fp_tensor.cpu().numpy().view(numpy_dtype), int_tensor, rtol=0, atol=0)

            fp_tensor.zero_()
            self.assertEqual(fp_tensor, torch.zeros_like(fp_tensor), rtol=0, atol=0)

        for fp_tensor in generate_inputs():
            run_test(fp_tensor)

        # Test that requires_grad is dropped, because view(dtype) does not support backward
        if dtype is torch.double:
            t = make_tensor((5, 5, 5), device, torch.double, low=-5, high=5, requires_grad=True)
            self.assertFalse(t.view(torch.complex64).requires_grad)


    @onlyOnCPUAndCUDA
    def test_view_as_complex(self, device):
        def fn(contiguous_input=True, dim0=0, dim1=1):
            t = torch.randn(3, 2, 2, device=device)
            c_t = t[:, :, 0] + 1j * t[:, :, 1]

            input = self._do_transpose(t, contiguous_input, dim0, dim1)

            if input.size()[-1] != 2:
                self.assertRaisesRegex(
                    RuntimeError, "Tensor must have a last dimension of size 2",
                    lambda: torch.view_as_complex(input))
                return

            if input.stride()[-1] != 1:
                self.assertRaisesRegex(
                    RuntimeError, "Tensor must have a last dimension with stride 1",
                    lambda: torch.view_as_complex(input))
                return

            res = torch.view_as_complex(input)
            self.assertEqual(res, self._do_transpose(c_t, contiguous_input, dim0, dim1))
            self.assertTrue(self.is_view_of(t, res))

        fn()
        fn(contiguous_input=False)
        # RuntimeError since in this case the last dim of input would not be of size 2
        fn(contiguous_input=False, dim0=0, dim1=2)
        # RuntimeError since in this case the last dim of input would not have stride 1
        fn(contiguous_input=False, dim0=1, dim1=2)


        # RuntimeError since in this case the stride of non-last dim of input would not be of size 2
        x = torch.randn(3, 3, device=device)
        t = torch.as_strided(x, (2, 2), (1, 1))
        self.assertRaisesRegex(
            RuntimeError, "Tensor must have a stride divisible by 2 for all but last dimension",
            lambda: torch.view_as_complex(t))

        # tensor with zero elements
        x = torch.tensor([], device=device)  # torch.Size([0])
        self.assertRaisesRegex(
            RuntimeError, "Tensor must have a last dimension of size 2",
            lambda: torch.view_as_complex(x))

        # zero dimension tensor
        z = torch.tensor(2.0)
        self.assertRaisesRegex(
            RuntimeError, "Input tensor must have one or more dimensions",
            lambda: torch.view_as_complex(z))

        y = x.reshape(0, 2)  # torch.Size([0, 2])
        res = torch.view_as_complex(y)
        self.assertTrue(self.is_view_of(x, res))
        self.assertEqual(res.shape, torch.Size([0]))

    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_complex_dtypes(include_complex32=True))
    def test_view_as_real(self, device, dtype):
        def fn(contiguous_input=True):
            t = torch.randn(3, 4, dtype=dtype, device=device)
            input = self._do_transpose(t, contiguous_input)
            res = torch.view_as_real(input)
            self.assertEqual(res[:, :, 0], input.real)
            self.assertEqual(res[:, :, 1], input.imag)
            # TODO: Add torch.ComplexHalfStorage
            if dtype != torch.complex32:
                self.assertTrue(self.is_view_of(t, res))
            else:
                self.assertRaises(RuntimeError, lambda: self.is_view_of(t, res))

        fn()
        fn(contiguous_input=False)

        # tensor with zero elements
        x = torch.tensor([], dtype=dtype, device=device)
        res = torch.view_as_real(x)
        # TODO: Add torch.ComplexHalfStorage
        if dtype != torch.complex32:
            self.assertTrue(self.is_view_of(x, res))
        else:
            self.assertRaises(RuntimeError, lambda: self.is_view_of(x, res))
        self.assertEqual(res.shape, torch.Size([0, 2]))

        # tensor with zero dim
        x = torch.tensor(2 + 3j, dtype=dtype, device=device)
        res = torch.view_as_real(x)
        # TODO: Add torch.ComplexHalfStorage
        if dtype != torch.complex32:
            self.assertTrue(self.is_view_of(x, res))
        else:
            self.assertRaises(RuntimeError, lambda: self.is_view_of(x, res))
        self.assertEqual(res.shape, torch.Size([2]))

    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_dtypes())
    def test_view_tensor_split(self, device, dtype):
        a = make_tensor((40, 30), device, dtype, low=-9, high=9)
        a_split_dim0 = a.tensor_split(7, 0)
        for a_split_dim0_tensor in a_split_dim0:
            self.assertTrue(self.is_view_of(a, a_split_dim0_tensor))
        a_split_dim1 = a.tensor_split(7, 1)
        for a_split_dim1_tensor in a_split_dim1:
            self.assertTrue(self.is_view_of(a, a_split_dim1_tensor))

    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_dtypes())
    def test_view_tensor_hsplit(self, device, dtype):
        t = make_tensor((4, 4, 4), device, dtype, low=-9, high=9)
        t_hsplit = torch.hsplit(t, 2)
        for t_hsplit_tensor in t_hsplit:
            self.assertTrue(self.is_view_of(t, t_hsplit_tensor))
        t[2, 2, 2] = 7
        self.assertEqual(t_hsplit[1][2, 0, 2], t[2, 2, 2])

    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_dtypes())
    def test_view_tensor_vsplit(self, device, dtype):
        t = make_tensor((4, 4, 4), device, dtype, low=-9, high=9)
        t_vsplit = torch.vsplit(t, 2)
        for t_vsplit_tensor in t_vsplit:
            self.assertTrue(self.is_view_of(t, t_vsplit_tensor))
        t[2, 2, 2] = 7
        self.assertEqual(t_vsplit[1][0, 2, 2], t[2, 2, 2])

    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_dtypes())
    def test_view_tensor_dsplit(self, device, dtype):
        t = make_tensor((4, 4, 4), device, dtype, low=-9, high=9)
        t_dsplit = torch.dsplit(t, 2)
        for t_dsplit_tensor in t_dsplit:
            self.assertTrue(self.is_view_of(t, t_dsplit_tensor))
        t[2, 2, 2] = 7
        self.assertEqual(t_dsplit[1][2, 2, 0], t[2, 2, 2])

    @onlyOnCPUAndCUDA
    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes()))
    def test_real_imag_noncomplex(self, device, dtype):
        t = torch.ones((5, 5), dtype=dtype, device=device)

        with self.assertRaises(RuntimeError):
            torch.real(t)

        with self.assertRaises(RuntimeError):
            torch.imag(t)

    @onlyOnCPUAndCUDA
    @dtypes(*torch.testing.get_all_complex_dtypes())
    def test_real_imag_view(self, device, dtype):
        def compare_with_numpy(contiguous_input=True):
            t = torch.randn(3, 3, dtype=dtype, device=device)
            if not contiguous_input:
                u = t.T
            else:
                u = t

            re = u.real
            exp = torch.from_numpy(u.cpu().numpy().real).to(device=device)
            self.assertEqual(re, exp)
            # for the case of contiguous_input, t=u
            # for the case of non contiguous_input, the base still remains
            # t since we are performing a view operation to make the input non-contiguous
            self.assertTrue(self.is_view_of(t, re))

            im = u.imag
            exp = torch.from_numpy(u.cpu().numpy().imag).to(device=device)
            self.assertEqual(im, exp)
            self.assertTrue(self.is_view_of(t, im))

        compare_with_numpy()
        compare_with_numpy(contiguous_input=False)

        # ensure storage offset is being correctly set
        a = torch.randn(10, dtype=dtype)
        self.assertEqual(a[5:].real, a.real[5:])
        self.assertEqual(a[5:].imag, a.imag[5:])

    @onlyOnCPUAndCUDA
    @dtypes(*product(torch.testing.get_all_complex_dtypes(), torch.testing.get_all_dtypes()))
    @suppress_warnings
    def test_set_real_imag(self, device, dtypes):
        x = torch.randn(10, dtype=dtypes[0], device=device)

        new_real = _make_tensor((10,), dtypes[1], device)
        new_imag = _make_tensor((10,), dtypes[1], device)

        x.real = new_real
        x.imag = new_imag

        if dtypes[1].is_complex:
            self.assertEqual(x.real, new_real.real, exact_dtype=False)
            self.assertEqual(x.imag, new_imag.real, exact_dtype=False)

        else:
            self.assertEqual(x.real, new_real, exact_dtype=False)
            self.assertEqual(x.imag, new_imag, exact_dtype=False)

    def test_diagonal_view(self, device) -> None:
        t = torch.ones((5, 5), device=device)
        v = torch.diagonal(t)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[0, 0], v[0])

        t = torch.ones((3, 3, 3), device=device)
        v = torch.diagonal(t, offset=1, dim1=1, dim2=2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0, 1], v[0, 0])

    def test_select_view(self, device) -> None:
        t = torch.ones((5, 5), device=device)
        v = t.select(0, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[2, 0], v[0])

    def test_unbind_view(self, device) -> None:
        t = torch.zeros((5, 5), device=device)
        tup = torch.unbind(t)

        for idx, v in enumerate(tup):
            self.assertTrue(self.is_view_of(t, v))

            v[0] = idx + 1
            self.assertEqual(t[idx, 0], v[0])

    def test_expand_view(self, device) -> None:
        t = torch.ones((5, 1), device=device)
        v = t.expand(5, 5)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_expand_as_view(self, device):
        t = torch.ones((5, 1), device=device)
        e = torch.empty((5, 5), device=device)
        v = t.expand_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_narrow_view(self, device):
        t = torch.ones((5, 5), device=device)
        v = torch.narrow(t, 1, 2, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 2], v[0, 0])

    def test_permute_view(self, device) -> None:
        t = torch.ones((5, 5), device=device)
        v = t.permute(1, 0)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_transpose_view(self, device):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            t = torch.ones((5, 5), device=device)
            v = fn(t, 0, 1)
            self.assertTrue(self.is_view_of(t, v))

            v[0, 1] = 0
            self.assertEqual(t[1, 0], v[0, 1])

    def test_t_view(self, device):
        t = torch.ones((5, 5), device=device)
        v = t.t()
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_T_view(self, device):
        t = torch.ones((5, 5), device=device)
        v = t.T
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_unfold_view(self, device):
        t = torch.ones(10, device=device)
        v = t.unfold(0, 3, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[1, 0] = 0
        self.assertEqual(t[2], v[1, 0])

    def test_squeeze_view(self, device):
        t = torch.ones(5, 1, 5, device=device)
        v = torch.squeeze(t)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t, v._base)

    def test_unsqueeze_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = torch.unsqueeze(t, 1)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])

    def test_as_strided_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = torch.as_strided(t, (25,), (1,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view(25)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_as_view(self, device):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,))
        v = t.view_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_contiguous_self(self, device):
        t = torch.ones(5, 5, device=device)
        s = t.contiguous()
        self.assertTrue(s is t)

    def test_contiguous_nonview(self, device):
        t = torch.ones(5, 5, device=device)
        nv = t.t().contiguous()
        self.assertTrue(not self.is_view_of(t, nv))

        nv[0, 0] = 0
        self.assertNotEqual(t[0, 0], nv[0, 0])

    def test_reshape_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = torch.reshape(t, (25,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_reshape_as_view(self, device):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,), device=device)
        v = t.reshape_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_reshape_nonview(self, device):
        t = torch.ones(5, 5, device=device)
        nv = torch.reshape(t.t(), (25,))
        self.assertTrue(not self.is_view_of(t, nv))

        nv[6] = 0
        self.assertNotEqual(t[1, 1], nv[6])

    def test_flatten_view(self, device):
        def test_writes_propagate(t, v):
            idx_t = (0,) * t.ndim
            idx_v = (0,) * v.ndim
            v[idx_v] = 0
            self.assertEqual(t[idx_t], v[idx_v])

        t = torch.ones(1, 2, 3, 4, device=device)
        v = t.flatten()
        self.assertTrue(self.is_view_of(t, v))
        test_writes_propagate(t, v)

        # zero-dimensional tensor
        t = torch.tensor(1, device=device)
        v = t.flatten()
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of(t, v))

        t = torch.ones(1, 2, 3, 4, device=device).transpose(2, 3)
        v = t.flatten(0, 1)
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of_same_base(t, v))

        # stride[i] = stride[i + 1] * size[i + 1] is satisfied for 3 groups:
        t = torch.ones(720, device=device) \
            .as_strided((2, 3, 2, 3, 5, 4), (6, 2, 15, 5, 1, 0))
        #               [--1--|---2---|-3-] [--1--|----2---|-3-]
        v1 = t.flatten(0, 1)
        v2 = v1.flatten(1, 3)
        v3 = v2.flatten(2, 2)
        test_writes_propagate(t, v1)
        self.assertTrue(self.is_view_of_same_base(t, v1))
        test_writes_propagate(t, v2)
        self.assertTrue(self.is_view_of_same_base(t, v2))
        test_writes_propagate(t, v3)
        self.assertTrue(self.is_view_of_same_base(t, v3))

    @onlyOnCPUAndCUDA
    def test_flatten_nonview(self, device):
        def assert_is_nonview(t, nv):
            idx_t = (0,) * t.ndim
            idx_nv = (0,) * nv.ndim
            self.assertTrue(not nv._is_view())
            nv[idx_nv] = 0
            self.assertNotEqual(t[idx_t], nv[idx_nv])
        t = torch.ones(2, 3, 2, 3, device=device).transpose(2, 3)
        nv = t.flatten(1, 3)
        assert_is_nonview(t, nv)

        t = torch.ones(2, 2, device=device).T
        nv = t.flatten()
        assert_is_nonview(t, nv)

        # flatten returns the original object if start_dim=end_dim
        t = t = torch.ones(2, 2, device=device)
        nv = t.flatten(1, 1)
        self.assertTrue(t is nv)

    def test_basic_indexing_slice_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t[:2, :3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_ellipses_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t[..., :2]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_newaxis_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t[None, :2, 3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 3], v[0, 0])

    def test_advanced_indexing_nonview(self, device):
        t = torch.ones(3, 3, device=device)
        rows = torch.tensor([[0, 0], [2, 2]], device=device)
        cols = torch.tensor([[0, 1], [2, 2]], device=device)
        nv = t[rows, cols]
        self.assertTrue(not self.is_view_of(t, nv))

        nv[1, 1] = 0
        self.assertNotEqual(t[2, 2], nv[1, 1])

    def test_advanced_indexing_assignment(self, device):
        t = torch.ones(3, 3, device=device)
        rows = torch.tensor([[0, 0], [2, 2]], device=device)
        cols = torch.tensor([[0, 1], [2, 2]], device=device)
        t[rows, cols] = 0
        self.assertEqual(t[2, 2], 0)

    @unittest.skip("See https://github.com/pytorch/pytorch/pull/32720")
    def test_chunk_view(self, device):
        t = torch.zeros(3, 3, device=device)
        l = torch.chunk(t, 3)

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    @unittest.skip("See https://github.com/pytorch/pytorch/pull/32720")
    def test_split_view(self, device):
        t = torch.zeros(3, 3, device=device)
        l = torch.split(t, [1, 1, 1])

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    def test_movedim_view(self, device):
        def run_test(device, op):
            t = torch.zeros(3, 3, device=device)
            out = op(t)

            self.assertTrue(self.is_view_of(t, out))

            # Randomly change values in output
            # and verify that original is changed
            # as well.
            for _ in range(3):
                idx_1, idx_2 = random.randint(0, 2), random.randint(0, 2)
                out[idx_1, idx_2] = random.random()
                self.assertEqual(t[idx_2, idx_1], out[idx_1, idx_2])

        for fn in [torch.movedim, torch.moveaxis]:
            op = partial(fn, source=(0, 1), destination=(1, 0))
            run_test(device, op)

            op = partial(fn, source=0, destination=1)
            run_test(device, op)

class TestOldViewOps(TestCase):
    def test_ravel(self, device):

        def _test_ravel(tensors, size, nc=False):
            for src in tensors:
                # Continuous Tensor -> View
                flat = src.ravel()
                self.assertEqual(flat.shape, torch.Size([size]))
                self.assertEqual(src.view(-1), flat)
                self.assertEqual(flat._base, src)

                # Non-continuous Tensor -> Copy
                if nc:
                    nc_src = src.t()
                    nc_flat = nc_src.ravel()
                    self.assertEqual(nc_flat.shape, torch.Size([size]))
                    self.assertEqual(nc_src.reshape(-1), nc_flat)
                    self.assertTrue(nc_flat._base != nc_src)

        # Test that flatten returns 1-dim tensor when given a 0-dim tensor
        zero_dim_tensor = torch.tensor(123, device=device)
        flat0 = zero_dim_tensor.ravel()
        one_dim_tensor = torch.tensor([123], device=device)
        flat1 = zero_dim_tensor.ravel()

        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        self.assertEqual(flat0.shape, torch.Size([1]))
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        self.assertEqual(flat1.shape, torch.Size([1]))
        self.assertEqual(flat0, one_dim_tensor)
        self.assertEqual(flat0, flat1)
        self.assertEqual(flat0.shape, flat1.shape)

        # Test both float tensor and quantized tensor
        tensors = [torch.randn(5, 5, 5, 5, device=device),
                   torch._empty_affine_quantized([5, 5, 5, 5],
                                                 scale=2,
                                                 zero_point=3,
                                                 dtype=torch.quint8,
                                                 device=device)]
        _test_ravel(tensors, 625)

        tensors = [torch.randn(0, 2, 3, device=device),
                   torch.randn(3, 0, 2, device=device),
                   torch._empty_affine_quantized([0, 2, 3],
                                                 scale=2,
                                                 zero_point=3,
                                                 dtype=torch.quint8,
                                                 device=device),
                   torch._empty_affine_quantized([3, 0, 2],
                                                 scale=2,
                                                 zero_point=3,
                                                 dtype=torch.quint8,
                                                 device=device)]
        _test_ravel(tensors, 0)

        tensors = [torch.randn(5, 5, device=device),
                   torch._empty_affine_quantized([5, 5],
                                                 scale=2,
                                                 zero_point=3,
                                                 dtype=torch.quint8,
                                                 device=device)]
        _test_ravel(tensors, 25, True)

    # TODO: this should be refactored into the view ops test suite
    def test_empty_reshape(self, device):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.reshape(1, 0, 6, 1, 1).shape)
        # should be viewable -- i.e. data_ptr is the same.
        self.assertEqual(x.data_ptr(), x.reshape(1, 0, 6, 1, 1).data_ptr())

        # match NumPy semantics -- don't infer the size of dimension with a degree of freedom
        self.assertRaises(RuntimeError, lambda: x.reshape(0, -1))

    def test_expand(self, device):
        tensor = torch.rand(1, 8, 1, device=device)
        tensor2 = torch.rand(5, device=device)
        template = torch.rand(4, 8, 5, device=device)
        target = template.size()
        self.assertEqual(tensor.expand_as(template).size(), target)
        self.assertEqual(tensor.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor.expand(target).size(), target)
        self.assertEqual(tensor2.expand_as(template).size(), target)
        self.assertEqual(tensor2.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor2.expand(target).size(), target)

        # test double expand
        self.assertEqual(tensor2.expand(1, 5).expand(2, 2, 5), tensor2.repeat(2, 2, 1))

        # test non-contiguous
        noncontig = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        self.assertFalse(noncontig.is_contiguous())
        self.assertEqual(noncontig.expand(2, 5, 4, 3), noncontig.contiguous().repeat(2, 1, 4, 1))

        # make sure it's compatible with unsqueeze
        expanded = tensor2.expand(1, 1, 5)
        unsqueezed = tensor2.unsqueeze(0).unsqueeze(1)
        self.assertEqual(expanded, unsqueezed)
        self.assertEqual(expanded.stride(), unsqueezed.stride())

        # test -1 as target size
        self.assertEqual(tensor.expand(4, -1, 5), tensor.expand(4, 8, 5))
        self.assertRaises(RuntimeError, lambda: tensor2.expand(-1, -1))

        # test expanding empty to empty
        self.assertEqual(torch.zeros(0, device=device).expand((0,)), torch.zeros(0, device=device))

    # TODO: this should be refactored into the view ops test suite
    def test_view_empty(self, device):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.view(1, 0, 6, 1, 1).shape)

    # TODO: this should be refactored into the view ops test suite
    @onlyOnCPUAndCUDA
    def test_reshape(self, device):
        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape(-1).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape(1, 9, 1).data_ptr())
        self.assertEqual(torch.reshape(x, (9,)), x.reshape(9))
        self.assertRaises(RuntimeError, lambda: x.reshape(-1, -1))

        y = torch.randn(4, 4, 4, device=device)[:, 0, :]
        self.assertNotEqual(y.data_ptr(), y.reshape(-1).data_ptr())
        self.assertEqual(y.contiguous().view(-1), y.reshape(-1))
        self.assertEqual(y.reshape(2, 2, 4).data_ptr(), y.data_ptr())

        s = torch.randn((), device=device)
        self.assertEqual(s.data_ptr(), s.reshape(()).data_ptr())
        self.assertEqual(s.reshape(-1).shape, (1,))
        self.assertRaises(RuntimeError, lambda: s.reshape(2))

        empty = torch.tensor([], device=device)
        self.assertEqual(empty, empty.reshape(-1))
        self.assertEqual(empty, empty.reshape([0]))
        # TODO: fix these once we have multi-dimensional empty tensors
        self.assertEqual(empty.reshape([0, 1]).shape, (0, 1))
        self.assertEqual(empty.reshape([1, -1]).shape, (1, 0))
        self.assertRaises(RuntimeError, lambda: empty.reshape(1))

        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(9)).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(1, 9, 1)).data_ptr())
        self.assertRaises(RuntimeError, lambda: x.reshape_as(torch.rand(10, device=device)))

    def test_flatten(self, device):
        # Test that flatten returns 1-dim tensor when given a 0-dim tensor
        zero_dim_tensor = torch.tensor(123, device=device)
        flat0 = zero_dim_tensor.flatten()
        one_dim_tensor = torch.tensor([123], device=device)
        flat1 = zero_dim_tensor.flatten()

        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        self.assertEqual(flat0.shape, torch.Size([1]))
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        self.assertEqual(flat1.shape, torch.Size([1]))
        self.assertEqual(flat0, one_dim_tensor)
        self.assertEqual(flat0, flat1)
        self.assertEqual(flat0.shape, flat1.shape)

        # Test both float tensor and quantized tensor
        tensors = [torch.randn(5, 5, 5, 5, device=device),
                   torch._empty_affine_quantized([5, 5, 5, 5],
                                                 scale=2,
                                                 zero_point=3,
                                                 dtype=torch.quint8,
                                                 device=device)]
        for src in tensors:
            flat = src.flatten(0, -1)
            self.assertEqual(flat.shape, torch.Size([625]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 2)
            self.assertEqual(flat.shape, torch.Size([125, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 1)
            self.assertEqual(flat.shape, torch.Size([25, 5, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(1, 2)
            self.assertEqual(flat.shape, torch.Size([5, 25, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 3)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(-2, -1)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 2)
            self.assertEqual(flat, src)

            # out of bounds index
            with self.assertRaisesRegex(IndexError, 'Dimension out of range'):
                src.flatten(5, 10)

            # invalid start and end
            with self.assertRaisesRegex(RuntimeError, 'start_dim cannot come after end_dim'):
                src.flatten(2, 0)

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_narrow(self, device):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, 0, 1), torch.tensor([[0, 1, 2]]))
        self.assertEqual(x.narrow(0, 0, 2), torch.tensor([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(x.narrow(0, 1, 1), torch.tensor([[3, 4, 5]]))
        self.assertEqual(x.narrow(0, -1, 1), torch.tensor([[6, 7, 8]]))
        self.assertEqual(x.narrow(0, -2, 2), torch.tensor([[3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(0, -3, 3), torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        self.assertEqual(x.narrow(-1, -1, 1), torch.tensor([[2], [5], [8]]))
        self.assertEqual(x.narrow(-2, -1, 1), torch.tensor([[6, 7, 8]]))

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_narrow_tensor(self, device):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, torch.tensor(0), 1), torch.tensor([[0, 1, 2]]))
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor(0.), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0]), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0, 1]), 1)

    # TODO: make work on CUDA, too
    @onlyCPU
    def test_t(self, device):
        # Test 0D tensors
        x = torch.randn(())
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 1D tensors
        x = torch.arange(4)
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 2D tensors
        x = torch.rand((2, 2))
        self.assertEqual(x.t(), x.transpose(0, 1))
        x = x.to_sparse()
        self.assertEqual(x.t(), x.transpose(0, 1))

        # Test 3D tensor
        x = torch.rand((2, 2, 2))
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 dimensions, but self is 3D'):
            x.t()
        x = x.to_sparse()
        with self.assertRaisesRegex(RuntimeError, 'expects a tensor with <= 2 sparse and 0 dense dimensions'):
            x.t()

    @onlyCPU
    def test_split(self, device):
        tensor = torch.rand(7, 4)
        split_size = 3
        dim = 0
        target_sizes = ([3, 4], [3, 4], [1, 4])
        splits = tensor.split(split_size, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        # Variable sections split
        tensor = torch.randn(20, 10)
        dim = 0
        split_sizes = [5, 5, 10]
        target_sizes = ([[5, 10], [5, 10], [10, 10]])
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

        split_sizes = [2, 2, 6]
        target_sizes = ([20, 2], [20, 2], [20, 6])
        dim = 1
        splits = tensor.split(split_sizes, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, atol=0, rtol=0)
            start = start + target_size[dim]

    @onlyCPU
    def test_chunk(self, device):
        tensor = torch.rand(4, 7)
        num_chunks = 3
        dim = 1
        target_sizes = ([4, 3], [4, 3], [4, 1])
        splits = tensor.chunk(num_chunks, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split,
                             atol=0, rtol=0)
            start = start + target_size[dim]

        # Invalid chunk sizes
        error_regex = 'chunk expects.*greater than 0'
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(0)
        with self.assertRaisesRegex(RuntimeError, error_regex):
            tensor.chunk(-2)

    # TODO: make work on CUDA, too
    @onlyCPU
    def test_unsqueeze(self, device) -> None:
        x = torch.randn(2, 3, 4)
        y = x.unsqueeze(1)
        self.assertEqual(y, x.view(2, 1, 3, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.view(2, 3, 1, 4))

        x = x[:, 1]
        self.assertFalse(x.is_contiguous())
        y = x.unsqueeze(1)
        self.assertEqual(y, x.contiguous().view(2, 1, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.contiguous().view(2, 4, 1))

    # unit test for special case transposed copy (see ATen/native/Copy.cpp for details)
    def test_big_transpose(self, device):
        t = torch.rand(456, 789, device=device)
        t1 = t.t().contiguous()
        t2 = torch.from_numpy(t.cpu().numpy().transpose())
        self.assertEqual(t1, t2)

    def test_T(self, device):
        a = torch.randn(2, 3, 4, device=device)
        t1 = a.T
        t2 = a.permute(2, 1, 0)
        self.assertEqual(t2, t1)
        b = torch.randn(10, device=device)
        self.assertEqual(b, b.T)
        scalar = torch.tensor(5, device=device)
        self.assertEqual(scalar, scalar.T)

    def test_python_types(self, device):
        a1 = torch.randn((1, 2), device=device, dtype=torch.float64)
        a2 = torch.randn((1, 2), device=device, dtype=float)
        self.assertEqual(a1.dtype, a2.dtype)

        b1 = torch.arange(10, 20, dtype=torch.int64, device=device)
        b2 = torch.arange(10, 20, dtype=int, device=device)
        self.assertEqual(b1.dtype, b2.dtype)

        c1 = torch.tensor([True, False], dtype=torch.bool, device=device)
        c2 = torch.tensor([True, False], dtype=bool, device=device)
        self.assertEqual(c1.dtype, c2.dtype)

    # TODO: is resize best put in test_view_ops?
    def test_resize_as_preserves_strides(self, device):
        x = torch.empty(2, 3).t()
        old_strides = x.stride()
        x.resize_as_(x)
        self.assertEqual(x.stride(), old_strides)

    def test_memory_format_resize_as(self, device):
        def test_helper(shape, memory_format, device):
            xc = torch.randn(shape, device=device).contiguous(memory_format=memory_format)
            flat = torch.randn(xc.numel(), device=device)
            flat.resize_as_(xc, memory_format=torch.preserve_format)
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        test_helper((10, 3, 32, 32), torch.channels_last, device)
        test_helper((3, 10, 3, 32, 32), torch.channels_last_3d, device)

    def test_memory_format_resize_(self, device):
        def test_helper(shape, numel, memory_format, device):
            flat = torch.randn(numel, device=device)
            flat.resize_(shape, memory_format=memory_format)
            self.assertTrue(flat.is_contiguous(memory_format=memory_format))

        test_helper((10, 3, 32, 32), 10 * 3 * 32 * 32, torch.channels_last, device)
        test_helper((3, 10, 3, 32, 32), 3 * 10 * 3 * 32 * 32, torch.channels_last_3d, device)

    @onlyOnCPUAndCUDA
    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_transpose_invalid(self, device, dtype):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            shape = _rand_shape(4, min_size=5, max_size=10)
            x = _generate_input(shape, dtype, device, False)

            # Invalid `source` and `destination` dimension
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_transpose_vs_numpy(self, device, dtype):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            for nd in range(5):
                shape = _rand_shape(nd, min_size=5, max_size=10)
                x = _generate_input(shape, dtype, device, with_extremal=False)
                for random_negative in [True, False]:
                    for src_dim, dst_dim in permutations(range(nd), r=2):
                        random_prob = random.random()

                        if random_negative and random_prob > 0.66:
                            src_dim = src_dim - nd
                        elif random_negative and random_prob > 0.33:
                            dst_dim = dst_dim - nd
                        elif random_negative:
                            src_dim = src_dim - nd
                            dst_dim = dst_dim - nd

                        partial_map = {
                            torch.swapdims: partial(torch.swapdims, dim0=src_dim, dim1=dst_dim),
                            torch.swapaxes: partial(torch.swapaxes, axis0=src_dim, axis1=dst_dim),
                            torch.transpose: partial(torch.transpose, dim0=src_dim, dim1=dst_dim),
                        }

                        torch_fn = partial_map[fn]
                        np_fn = partial(np.swapaxes, axis1=src_dim, axis2=dst_dim)
                        self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            # Move dim to same position
            x = torch.randn(2, 3, 5, 7, 11)
            partial_map = {
                torch.swapdims: partial(torch.swapdims, dim0=0, dim1=0),
                torch.swapaxes: partial(torch.swapaxes, axis0=0, axis1=0),
                torch.transpose: partial(torch.transpose, dim0=0, dim1=0),
            }
            torch_fn = partial_map[fn]
            np_fn = partial(np.swapaxes, axis1=0, axis2=0)
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

    def _test_atleast_dim(self, torch_fn, np_fn, device, dtype):
        for ndims in range(0, 5):
            shape = _rand_shape(ndims, min_size=5, max_size=10)
            for n in range(ndims + 1):
                for with_extremal in [False, True]:
                    for contiguous in [False, True]:
                        # Generate Input.
                        x = _generate_input(shape, dtype, device, with_extremal)
                        if contiguous:
                            x = x.T
                        self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

                        # Compare sequence input
                        torch_sequence_x = (x,) * random.randint(3, 10)
                        np_sequence_x = tuple(np.array(x.detach().cpu().numpy()) for x in torch_sequence_x)
                        torch_res = torch_fn(*torch_sequence_x)
                        np_res = np_fn(*np_sequence_x)

                        torch_res = tuple(x.cpu() for x in torch_res)
                        np_res = tuple(torch.from_numpy(x) for x in np_res)
                        self.assertEqual(np_res, torch_res)

    # TODO: are these view ops?
    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False) +
              torch.testing.get_all_complex_dtypes()))
    def test_atleast(self, device, dtype):
        self._test_atleast_dim(torch.atleast_1d, np.atleast_1d, device, dtype)
        self._test_atleast_dim(torch.atleast_2d, np.atleast_2d, device, dtype)
        self._test_atleast_dim(torch.atleast_3d, np.atleast_3d, device, dtype)

    @onlyCPU
    @dtypes(torch.float)
    def test_broadcast_tensors(self, device, dtype):
        x0 = torch.randn(2, 1, 3, dtype=dtype, device=device)
        x1 = torch.randn(3, dtype=dtype, device=device)
        x2 = torch.randn(3, 1, dtype=dtype, device=device)
        expected_size = (2, 3, 3)

        y0, y1, y2 = torch.broadcast_tensors(x0, x1, x2)
        self.assertTrue(y0.size() == expected_size)
        self.assertTrue(y1.size() == expected_size)
        self.assertTrue(y2.size() == expected_size)


    @onlyCPU
    def test_broadcast_shapes(self, device):
        examples = [(), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)]
        for s0 in examples:
            x0 = torch.randn(s0)
            expected = torch.broadcast_tensors(x0)[0].shape
            actual = torch.broadcast_shapes(s0)
            self.assertEqual(expected, actual)

            for s1 in examples:
                x1 = torch.randn(s1)
                expected = torch.broadcast_tensors(x0, x1)[0].shape
                actual = torch.broadcast_shapes(s0, s1)
                self.assertEqual(expected, actual)

    # Skip BFloat16 since numpy does not support it
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False))
    def test_broadcast_to(self, device, dtype):
        def can_broadcast(s0, s1):
            # s0.dim() <= s1.dim(), reverse s0 and s1 to compare trailing dimension
            s0 = tuple(reversed(s0))
            s1 = tuple(reversed(s1))
            for i in range(len(s0)):
                if s0[i] != 1 and s0[i] != s1[i]:
                    return False
            return True

        sizes = (
            (), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)
        )
        for s0, s1 in combinations(sizes, r=2):
            t = make_tensor(s0, device, dtype, low=-9, high=9)
            t_np = t.cpu().numpy()

            if can_broadcast(s0, s1):
                res = torch.broadcast_to(t, s1)
                np_res = np.broadcast_to(t_np, s1)
                self.assertEqual(res, np_res)
            else:
                with self.assertRaisesRegex(RuntimeError,
                                            r"The expanded size of the tensor \(\d\) "
                                            r"must match the existing size \(\d\)"):
                    torch.broadcast_to(t, s1)

    def test_view(self, device):
        tensor = torch.rand(15, device=device)
        template = torch.rand(3, 5, device=device)
        empty = torch.empty(0, device=device)
        target = template.size()
        self.assertEqual(tensor.view_as(template).size(), target)
        self.assertEqual(tensor.view(3, 5).size(), target)
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        self.assertEqual(tensor.view(-1, 5).size(), target)
        self.assertEqual(tensor.view(3, -1).size(), target)
        tensor_view = tensor.view(5, 3)
        tensor_view.fill_(random.uniform(0, 1))
        self.assertEqual(empty.view_as(empty), empty)
        self.assertEqual(empty.view(0), empty)
        self.assertEqual(empty.view(0, 3, 0, 1).size(), torch.Size([0, 3, 0, 1]))
        self.assertEqual(empty.view(0, 3, 0, 1).view(0), empty)

        # test size inference with empty tensors
        self.assertEqual(empty.view(-1).size(), torch.Size([0]))
        self.assertEqual(empty.view(10, 3, -1).size(), torch.Size([10, 3, 0]))

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(-1, 0)

        with self.assertRaisesRegex(RuntimeError, r"because the unspecified dimension size -1 can be any value"):
            empty.view(3, 0, -1, 0)

        self.assertRaises(RuntimeError, lambda: tensor.view(15, 0))
        self.assertRaises(RuntimeError, lambda: tensor.view(7, -1))
        self.assertRaises(RuntimeError, lambda: tensor.view(15, -1, -1))

        # test view when tensor is not contiguous in every dimension, but only
        # contiguous dimensions are touched.
        tensor = torch.rand(4, 2, 5, 1, 6, 2, 9, 3, device=device).transpose(-1, 2).transpose(-2, 3)
        # size:                      [   4,    2,    3,    9,    6,    2,    1,    5]
        # stride:                    [3840, 1620,    1,    3,   54,   27,  324,  324]
        # contiguous dim chunks:     [__________, ____, ____, __________, ____, ____]
        # merging 1 to chunk after:  [__________, ____, ____, __________, __________]
        contig_tensor = tensor.clone()
        # [4, 2] => [8, 1]
        # [3] => [3]
        # [9] => [3, 3]
        # [6, 2] => [4, 1, 3]
        # [1, 5] => [5]
        view_size = [8, 1, 3, 3, 3, 4, 1, 3, 5]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))
        # [4, 2] => [2, 4]
        # [3] => [3]
        # [9] => [1, 9]
        # [6, 2] => [2, 2, 3]
        # [1, 5] => [5, 1]
        view_size = [2, 4, 3, 1, 9, 2, 2, 3, 5, 1]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))
        # adding size 1 dims
        view_size = [1, 1, 2, 1, 4, 3, 1, 1, 9, 1, 2, 1, 2, 3, 1, 5, 1, 1]
        self.assertEqual(tensor.view(*view_size), contig_tensor.view(*view_size))

        # invalid views
        self.assertRaises(RuntimeError, lambda: tensor.view(-1))
        # crossing [4, 2], [3]
        self.assertRaises(RuntimeError, lambda: tensor.view(24, 9, 6, 2, 1, 5))
        # crossing [6, 2], [1, 5]
        self.assertRaises(RuntimeError, lambda: tensor.view(8, 3, 9, 6, 10))
        # crossing [9], [6, 2]
        self.assertRaises(RuntimeError, lambda: tensor.view(8, 3, 54, 2, 1, 5))

        # view with stride 0 dims
        tensor = torch.empty(1, 1, device=device).expand(3, 4)  # all dims are contiguous
        contig_tensor = tensor.clone()
        self.assertEqual(tensor.view(-1), contig_tensor.view(-1))
        self.assertEqual(tensor.view(1, -1, 1), contig_tensor.view(1, -1, 1))
        self.assertEqual(tensor.view(-1, 1), contig_tensor.view(-1, 1))
        self.assertEqual(tensor.view(6, 2, 1), contig_tensor.view(6, 2, 1))
        self.assertEqual(tensor.view(1, 6, 2, 1), contig_tensor.view(1, 6, 2, 1))

    def test_contiguous(self, device):
        x = torch.randn(1, 16, 5, 5, device=device)
        self.assertTrue(x.is_contiguous())
        stride = list(x.stride())
        stride[0] = 20
        # change the stride in dimension 0. the tensor is still contiguous because size[0] is 1
        x.set_(x.storage(), 0, x.size(), stride)
        self.assertTrue(x.is_contiguous())

    @onlyOnCPUAndCUDA
    # Skip BFloat16 since numpy does not support it
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False))
    def test_tensor_split_sections(self, device, dtype):
        input_sizes = [
            (0,),
            (10,),
            (10, 0),
            (0, 10),
            (4, 10),
            (12, 3),
        ]
        for input_size in input_sizes:
            a_base = make_tensor(input_size, device, dtype, low=-9, high=9)
            # Run tests on transposed input if it has at least 2 dims
            for a in [a_base, a_base.t()] if a_base.dim() > 2 else [a_base]:
                a_n = a.cpu().numpy()
                for dim in range(-a.dim(), a.dim()):
                    for sections in range(1, 2 * a.size(dim)):
                        msg = f'input_size {input_size}, sections {sections}, dim {dim}'
                        result1 = torch.tensor_split(a, sections, dim)
                        result2 = torch.tensor_split(a, torch.tensor(sections, dtype=torch.int64), dim)
                        for r1, r2 in zip(result1, result2):
                            self.assertEqual(r1.device, torch.device(device), msg=msg)
                            self.assertEqual(r1.dtype, dtype, msg=msg)
                            self.assertEqual(r2.device, torch.device(device), msg=msg)
                            self.assertEqual(r2.dtype, dtype, msg=msg)
                        result_n = np.array_split(a_n, sections, dim)
                        self.assertEqual(result_n, result1, msg=msg)
                        self.assertEqual(result_n, result2, msg=msg)

    @onlyOnCPUAndCUDA
    # Skip BFloat16 since numpy does not support it
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False))
    def test_tensor_split_indices(self, device, dtype):
        input_sizes = [
            (0,),
            (10,),
            (10, 0),
            (0, 10),
            (4, 10),
            (12, 3),
        ]
        indices_args = [
            (),
            (0,),
            (3,),
            (10,),
            (-1,),
            (-10,),
            (2, -1),
            (3, 4, 10),
            (0, -1, 0, 10),
            (1, 5, 2, 8),
        ]
        for input_size in input_sizes:
            a_base = make_tensor(input_size, device, dtype, low=-9, high=9)
            # Run tests on transposed input if it has at least 2 dims
            for a in [a_base, a_base.t()] if a_base.dim() > 2 else [a_base]:
                a_n = a.cpu().numpy()
                for dim in range(-a.dim(), a.dim()):
                    for indices in indices_args:
                        result_1 = torch.tensor_split(a, indices, dim)
                        result_2 = torch.tensor_split(a, torch.tensor(indices, dtype=torch.int64), dim)

                        msg = f'input_size {input_size}, indices {indices}, dim {dim}'
                        for r1, r2 in zip(result_1, result_2):
                            self.assertEqual(r1.device, torch.device(device), msg=msg)
                            self.assertEqual(r1.dtype, dtype, msg=msg)
                            self.assertEqual(r2.device, torch.device(device), msg=msg)
                            self.assertEqual(r2.dtype, dtype, msg=msg)

                        result_n = np.array_split(a_n, indices, dim)
                        self.assertEqual(result_n, result_1, msg=msg)
                        self.assertEqual(result_n, result_2, msg=msg)

    @onlyOnCPUAndCUDA
    def test_tensor_split_errors(self, device):
        S = 10
        test_cases = [
            # input size, sections or indices, dim, error type, error message, numpy error type
            [(S,), 10, 1, IndexError, r'Dimension out of range', IndexError],
            [(), 10, 0, RuntimeError, r'tensor_split expected at least a 1-dimensional tensor, '
                + 'but got a tensor with 0 dims', IndexError],
            [(S,), (10,), 1, IndexError, r'Dimension out of range', IndexError],
            [(), (10,), 0, RuntimeError, r'tensor_split expected at least a 1-dimensional tensor, '
                + 'but got a tensor with 0 dims', IndexError],
            [(S,), 0, 0, RuntimeError, r'number of sections must be larger than 0, got 0', ValueError],
            [(S,), -1, 0, RuntimeError, r'number of sections must be larger than 0, got -1', ValueError],
        ]
        for input_size, sections_or_indices, dim, err, err_msg, numpy_err in test_cases:
            a = torch.randn(input_size, device=device)
            msg = f'input_size {input_size}, sections_or_indices {sections_or_indices}, dim {dim}'
            with self.assertRaisesRegex(err, err_msg, msg=msg):
                torch.tensor_split(a, sections_or_indices, dim)
            with self.assertRaisesRegex(err, err_msg, msg=msg):
                torch.tensor_split(a, torch.tensor(sections_or_indices), dim)
            with self.assertRaises(numpy_err, msg=msg):
                np.array_split(a.cpu().numpy(), sections_or_indices, dim)

        # addtional tests for tensor_split with tensor_indices_or_sections
        with self.assertRaisesRegex(RuntimeError,
                                    r'tensor_split expected tensor_indices_or_sections to have dtype of long, but got Float'):
            torch.tensor_split(a, torch.tensor(1.1), dim)

        with self.assertRaisesRegex(RuntimeError,
                                    r'tensor_split expected tensor_indices_or_sections to be a'
                                    + ' zero-dimensional or one-dimensional tensor, but got a tensor with 2 dims'):
            torch.tensor_split(torch.rand(S, device=device), torch.tensor(((1,),)), 0)

    def test_resize_all_dtypes_and_devices(self, device):
        shape = (2, 2)
        for dt in torch.testing.get_all_dtypes():
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            x.resize_(shape)
            self.assertEqual(shape, x.shape)

    def test_resize_as_all_dtypes_and_devices(self, device):
        for dt in torch.testing.get_all_dtypes():
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            y = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dt, device=device)
            x.resize_as_(y)
            self.assertEqual(y.shape, x.shape)

    def test_view_all_dtypes_and_devices(self, device):
        for dt in torch.testing.get_all_dtypes():
            x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=dt, device=device)
            self.assertEqual(x.view(6).shape, [6])


instantiate_device_type_tests(TestViewOps, globals())
instantiate_device_type_tests(TestOldViewOps, globals())

if __name__ == '__main__':
    run_tests()
