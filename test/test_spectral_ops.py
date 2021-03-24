import torch
import unittest
import math
from contextlib import contextmanager
from itertools import product
import itertools

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY, TEST_LIBROSA, TEST_MKL)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, onlyOnCPUAndCUDA,
     skipCPUIfNoMkl, skipCUDAIfRocm, deviceCountAtLeast, onlyCUDA, OpDTypes,
     skipIf)
from torch.testing._internal.common_methods_invocations import spectral_funcs

from distutils.version import LooseVersion
from typing import Optional, List


if TEST_NUMPY:
    import numpy as np


if TEST_LIBROSA:
    import librosa


def _complex_stft(x, *args, **kwargs):
    # Transform real and imaginary components separably
    stft_real = torch.stft(x.real, *args, **kwargs, return_complex=True, onesided=False)
    stft_imag = torch.stft(x.imag, *args, **kwargs, return_complex=True, onesided=False)
    return stft_real + 1j * stft_imag


def _hermitian_conj(x, dim):
    """Returns the hermitian conjugate along a single dimension

    H(x)[i] = conj(x[-i])
    """
    out = torch.empty_like(x)
    mid = (x.size(dim) - 1) // 2
    idx = [slice(None)] * out.dim()
    idx_center = list(idx)
    idx_center[dim] = 0
    out[idx] = x[idx]

    idx_neg = list(idx)
    idx_neg[dim] = slice(-mid, None)
    idx_pos = idx
    idx_pos[dim] = slice(1, mid + 1)

    out[idx_pos] = x[idx_neg].flip(dim)
    out[idx_neg] = x[idx_pos].flip(dim)
    if (2 * mid + 1 < x.size(dim)):
        idx[dim] = mid + 1
        out[idx] = x[idx]
    return out.conj()


def _complex_istft(x, *args, **kwargs):
    # Decompose into Hermitian (FFT of real) and anti-Hermitian (FFT of imaginary)
    n_fft = x.size(-2)
    slc = (Ellipsis, slice(None, n_fft // 2 + 1), slice(None))

    hconj = _hermitian_conj(x, dim=-2)
    x_hermitian = (x + hconj) / 2
    x_antihermitian = (x - hconj) / 2
    istft_real = torch.istft(x_hermitian[slc], *args, **kwargs, onesided=True)
    istft_imag = torch.istft(-1j * x_antihermitian[slc], *args, **kwargs, onesided=True)
    return torch.complex(istft_real, istft_imag)


def _stft_reference(x, hop_length, window):
    r"""Reference stft implementation

    This doesn't implement all of torch.stft, only the STFT definition:

    .. math:: X(m, \omega) = \sum_n x[n]w[n - m] e^{-jn\omega}

    """
    n_fft = window.numel()
    X = torch.empty((n_fft, (x.numel() - n_fft + hop_length) // hop_length),
                    device=x.device, dtype=torch.cdouble)
    for m in range(X.size(1)):
        start = m * hop_length
        if start + n_fft > x.numel():
            slc = torch.empty(n_fft, device=x.device, dtype=x.dtype)
            tmp = x[start:]
            slc[:tmp.numel()] = tmp
        else:
            slc = x[start: start + n_fft]
        X[:, m] = torch.fft.fft(slc * window)
    return X


# Tests of functions related to Fourier analysis in the torch.fft namespace
class TestFFT(TestCase):
    exact_dtype = True

    @onlyOnCPUAndCUDA
    @ops([op for op in spectral_funcs if not op.ndimensional])
    def test_reference_1d(self, device, dtype, op):
        norm_modes = ((None, "forward", "backward", "ortho")
                      if LooseVersion(np.__version__) >= '1.20.0'
                      else (None, "ortho"))
        test_args = [
            *product(
                # input
                (torch.randn(67, device=device, dtype=dtype),
                 torch.randn(80, device=device, dtype=dtype),
                 torch.randn(12, 14, device=device, dtype=dtype),
                 torch.randn(9, 6, 3, device=device, dtype=dtype)),
                # n
                (None, 50, 6),
                # dim
                (-1, 0),
                # norm
                norm_modes
            ),
            # Test transforming middle dimensions of multi-dim tensor
            *product(
                (torch.randn(4, 5, 6, 7, device=device, dtype=dtype),),
                (None,),
                (1, 2, -2,),
                norm_modes
            )
        ]

        for iargs in test_args:
            args = list(iargs)
            input = args[0]
            args = args[1:]

            expected = op.ref(input.cpu().numpy(), *args)
            exact_dtype = dtype in (torch.double, torch.complex128)
            actual = op(input, *args)
            self.assertEqual(actual, expected, exact_dtype=exact_dtype)

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_fft_round_trip(self, device, dtype):
        # Test that round trip through ifft(fft(x)) is the identity
        test_args = list(product(
            # input
            (torch.randn(67, device=device, dtype=dtype),
             torch.randn(80, device=device, dtype=dtype),
             torch.randn(12, 14, device=device, dtype=dtype),
             torch.randn(9, 6, 3, device=device, dtype=dtype)),
            # dim
            (-1, 0),
            # norm
            (None, "forward", "backward", "ortho")
        ))

        fft_functions = [(torch.fft.fft, torch.fft.ifft)]
        # Real-only functions
        if not dtype.is_complex:
            # NOTE: Using ihfft as "forward" transform to avoid needing to
            # generate true half-complex input
            fft_functions += [(torch.fft.rfft, torch.fft.irfft),
                              (torch.fft.ihfft, torch.fft.hfft)]

        for forward, backward in fft_functions:
            for x, dim, norm in test_args:
                kwargs = {
                    'n': x.size(dim),
                    'dim': dim,
                    'norm': norm,
                }

                y = backward(forward(x, **kwargs), **kwargs)
                # For real input, ifft(fft(x)) will convert to complex
                self.assertEqual(x, y, exact_dtype=(
                    forward != torch.fft.fft or x.is_complex()))

    # Note: NumPy will throw a ValueError for an empty input
    @onlyOnCPUAndCUDA
    @ops(spectral_funcs)
    def test_empty_fft(self, device, dtype, op):
        t = torch.empty(0, device=device, dtype=dtype)
        match = r"Invalid number of data points \([-\d]*\) specified"

        with self.assertRaisesRegex(RuntimeError, match):
            op(t)

    @onlyOnCPUAndCUDA
    def test_fft_invalid_dtypes(self, device):
        t = torch.randn(64, device=device, dtype=torch.complex128)

        with self.assertRaisesRegex(RuntimeError, "rfft expects a real input tensor"):
            torch.fft.rfft(t)

        with self.assertRaisesRegex(RuntimeError, "rfftn expects a real-valued input tensor"):
            torch.fft.rfftn(t)

        with self.assertRaisesRegex(RuntimeError, "ihfft expects a real input tensor"):
            torch.fft.ihfft(t)

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.int8, torch.float, torch.double, torch.complex64, torch.complex128)
    def test_fft_type_promotion(self, device, dtype):
        if dtype.is_complex or dtype.is_floating_point:
            t = torch.randn(64, device=device, dtype=dtype)
        else:
            t = torch.randint(-2, 2, (64,), device=device, dtype=dtype)

        PROMOTION_MAP = {
            torch.int8: torch.complex64,
            torch.float: torch.complex64,
            torch.double: torch.complex128,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
        }
        T = torch.fft.fft(t)
        self.assertEqual(T.dtype, PROMOTION_MAP[dtype])

        PROMOTION_MAP_C2R = {
            torch.int8: torch.float,
            torch.float: torch.float,
            torch.double: torch.double,
            torch.complex64: torch.float,
            torch.complex128: torch.double,
        }
        R = torch.fft.hfft(t)
        self.assertEqual(R.dtype, PROMOTION_MAP_C2R[dtype])

        if not dtype.is_complex:
            PROMOTION_MAP_R2C = {
                torch.int8: torch.complex64,
                torch.float: torch.complex64,
                torch.double: torch.complex128,
            }
            C = torch.fft.rfft(t)
            self.assertEqual(C.dtype, PROMOTION_MAP_R2C[dtype])

    @onlyOnCPUAndCUDA
    @ops(spectral_funcs, dtypes=OpDTypes.unsupported,
         allowed_dtypes=[torch.half, torch.bfloat16])
    def test_fft_half_and_bfloat16_errors(self, device, dtype, op):
        # TODO: Remove torch.half error when complex32 is fully implemented
        x = torch.randn(64, device=device).to(dtype)
        with self.assertRaisesRegex(RuntimeError, "Unsupported dtype "):
            op(x)

    # nd-fft tests

    @onlyOnCPUAndCUDA
    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    @ops([op for op in spectral_funcs if op.ndimensional])
    def test_reference_nd(self, device, dtype, op):
        norm_modes = ((None, "forward", "backward", "ortho")
                      if LooseVersion(np.__version__) >= '1.20.0'
                      else (None, "ortho"))

        # input_ndim, s, dim
        transform_desc = [
            *product(range(2, 5), (None,), (None, (0,), (0, -1))),
            *product(range(2, 5), (None, (4, 10)), (None,)),
            (6, None, None),
            (5, None, (1, 3, 4)),
            (3, None, (0, -1)),
            (3, None, (1,)),
            (1, None, (0,)),
            (4, (10, 10), None),
            (4, (10, 10), (0, 1))
        ]

        for input_ndim, s, dim in transform_desc:
            shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            input = torch.randn(*shape, device=device, dtype=dtype)
            for norm in norm_modes:
                expected = op.ref(input.cpu().numpy(), s, dim, norm)
                exact_dtype = dtype in (torch.double, torch.complex128)
                actual = op(input, s, dim, norm)
                self.assertEqual(actual, expected, exact_dtype=exact_dtype)

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_fftn_round_trip(self, device, dtype):
        norm_modes = (None, "forward", "backward", "ortho")

        # input_ndim, dim
        transform_desc = [
            *product(range(2, 5), (None, (0,), (0, -1))),
            *product(range(2, 5), (None,)),
            (7, None),
            (5, (1, 3, 4)),
            (3, (0, -1)),
            (3, (1,)),
            (1, 0),
        ]

        fft_functions = [(torch.fft.fftn, torch.fft.ifftn)]

        # Real-only functions
        if not dtype.is_complex:
            fft_functions += [(torch.fft.rfftn, torch.fft.irfftn)]

        for input_ndim, dim in transform_desc:
            shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            x = torch.randn(*shape, device=device, dtype=dtype)

            for (forward, backward), norm in product(fft_functions, norm_modes):
                if isinstance(dim, tuple):
                    s = [x.size(d) for d in dim]
                else:
                    s = x.size() if dim is None else x.size(dim)

                kwargs = {'s': s, 'dim': dim, 'norm': norm}
                y = backward(forward(x, **kwargs), **kwargs)
                # For real input, ifftn(fftn(x)) will convert to complex
                self.assertEqual(x, y, exact_dtype=(
                    forward != torch.fft.fftn or x.is_complex()))

    @onlyOnCPUAndCUDA
    @ops([op for op in spectral_funcs if op.ndimensional],
         allowed_dtypes=[torch.float, torch.cfloat])
    def test_fftn_invalid(self, device, dtype, op):
        a = torch.rand(10, 10, 10, device=device, dtype=dtype)

        with self.assertRaisesRegex(RuntimeError, "dims must be unique"):
            op(a, dim=(0, 1, 0))

        with self.assertRaisesRegex(RuntimeError, "dims must be unique"):
            op(a, dim=(2, -1))

        with self.assertRaisesRegex(RuntimeError, "dim and shape .* same length"):
            op(a, s=(1,), dim=(0, 1))

        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            op(a, dim=(3,))

        with self.assertRaisesRegex(RuntimeError, "tensor only has 3 dimensions"):
            op(a, s=(10, 10, 10, 10))

    # 2d-fft tests

    # NOTE: 2d transforms are only thin wrappers over n-dim transforms,
    # so don't require exhaustive testing.

    @skipCPUIfNoMkl
    @skipCUDAIfRocm
    @onlyOnCPUAndCUDA
    @dtypes(torch.double, torch.complex128)
    def test_fft2_numpy(self, device, dtype):
        norm_modes = ((None, "forward", "backward", "ortho")
                      if LooseVersion(np.__version__) >= '1.20.0'
                      else (None, "ortho"))

        # input_ndim, s
        transform_desc = [
            *product(range(2, 5), (None, (4, 10))),
        ]

        fft_functions = ['fft2', 'ifft2', 'irfft2']
        if dtype.is_floating_point:
            fft_functions += ['rfft2']

        for input_ndim, s in transform_desc:
            shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            input = torch.randn(*shape, device=device, dtype=dtype)
            for fname, norm in product(fft_functions, norm_modes):
                torch_fn = getattr(torch.fft, fname)
                numpy_fn = getattr(np.fft, fname)

                def fn(t: torch.Tensor, s: Optional[List[int]], dim: List[int] = (-2, -1), norm: Optional[str] = None):
                    return torch_fn(t, s, dim, norm)

                torch_fns = (torch_fn, torch.jit.script(fn))

                # Once with dim defaulted
                input_np = input.cpu().numpy()
                expected = numpy_fn(input_np, s, norm=norm)
                for fn in torch_fns:
                    actual = fn(input, s, norm=norm)
                    self.assertEqual(actual, expected)

                # Once with explicit dims
                dim = (1, 0)
                expected = numpy_fn(input.cpu(), s, dim, norm)
                for fn in torch_fns:
                    actual = fn(input, s, dim, norm)
                    self.assertEqual(actual, expected)

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.complex64)
    def test_fft2_fftn_equivalence(self, device, dtype):
        norm_modes = (None, "forward", "backward", "ortho")

        # input_ndim, s, dim
        transform_desc = [
            *product(range(2, 5), (None, (4, 10)), (None, (1, 0))),
            (3, None, (0, 2)),
        ]

        fft_functions = ['fft', 'ifft', 'irfft']
        # Real-only functions
        if dtype.is_floating_point:
            fft_functions += ['rfft']

        for input_ndim, s, dim in transform_desc:
            shape = itertools.islice(itertools.cycle(range(4, 9)), input_ndim)
            x = torch.randn(*shape, device=device, dtype=dtype)

            for func, norm in product(fft_functions, norm_modes):
                f2d = getattr(torch.fft, func + '2')
                fnd = getattr(torch.fft, func + 'n')

                kwargs = {'s': s, 'norm': norm}

                if dim is not None:
                    kwargs['dim'] = dim
                    expect = fnd(x, **kwargs)
                else:
                    expect = fnd(x, dim=(-2, -1), **kwargs)

                actual = f2d(x, **kwargs)

                self.assertEqual(actual, expect)

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    def test_fft2_invalid(self, device):
        a = torch.rand(10, 10, 10, device=device)
        fft_funcs = (torch.fft.fft2, torch.fft.ifft2,
                     torch.fft.rfft2, torch.fft.irfft2)

        for func in fft_funcs:
            with self.assertRaisesRegex(RuntimeError, "dims must be unique"):
                func(a, dim=(0, 0))

            with self.assertRaisesRegex(RuntimeError, "dims must be unique"):
                func(a, dim=(2, -1))

            with self.assertRaisesRegex(RuntimeError, "dim and shape .* same length"):
                func(a, s=(1,))

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                func(a, dim=(2, 3))

        c = torch.complex(a, a)
        with self.assertRaisesRegex(RuntimeError, "rfftn expects a real-valued input"):
            torch.fft.rfft2(c)

    # Helper functions

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    @dtypes(torch.float, torch.double)
    def test_fftfreq_numpy(self, device, dtype):
        test_args = [
            *product(
                # n
                range(1, 20),
                # d
                (None, 10.0),
            )
        ]

        functions = ['fftfreq', 'rfftfreq']

        for fname in functions:
            torch_fn = getattr(torch.fft, fname)
            numpy_fn = getattr(np.fft, fname)

            for n, d in test_args:
                args = (n,) if d is None else (n, d)
                expected = numpy_fn(*args)
                actual = torch_fn(*args, device=device, dtype=dtype)
                self.assertEqual(actual, expected, exact_dtype=False)

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double)
    def test_fftfreq_out(self, device, dtype):
        for func in (torch.fft.fftfreq, torch.fft.rfftfreq):
            expect = func(n=100, d=.5, device=device, dtype=dtype)
            actual = torch.empty((), device=device, dtype=dtype)
            with self.assertWarnsRegex(UserWarning, "out tensor will be resized"):
                func(n=100, d=.5, out=actual)
            self.assertEqual(actual, expect)


    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_fftshift_numpy(self, device, dtype):
        test_args = [
            # shape, dim
            *product(((11,), (12,)), (None, 0, -1)),
            *product(((4, 5), (6, 6)), (None, 0, (-1,))),
            *product(((1, 1, 4, 6, 7, 2),), (None, (3, 4))),
        ]

        functions = ['fftshift', 'ifftshift']

        for shape, dim in test_args:
            input = torch.rand(*shape, device=device, dtype=dtype)
            input_np = input.cpu().numpy()

            for fname in functions:
                torch_fn = getattr(torch.fft, fname)
                numpy_fn = getattr(np.fft, fname)

                expected = numpy_fn(input_np, axes=dim)
                actual = torch_fn(input, dim=dim)
                self.assertEqual(actual, expected)

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    @dtypes(torch.float, torch.double)
    def test_fftshift_frequencies(self, device, dtype):
        for n in range(10, 15):
            sorted_fft_freqs = torch.arange(-(n // 2), n - (n // 2),
                                            device=device, dtype=dtype)
            x = torch.fft.fftfreq(n, d=1 / n, device=device, dtype=dtype)

            # Test fftshift sorts the fftfreq output
            shifted = torch.fft.fftshift(x)
            self.assertTrue(torch.allclose(shifted, shifted.sort().values))
            self.assertEqual(sorted_fft_freqs, shifted)

            # And ifftshift is the inverse
            self.assertEqual(x, torch.fft.ifftshift(shifted))

    # Legacy fft tests
    def _test_fft_ifft_rfft_irfft(self, device, dtype):
        complex_dtype = {
            torch.float16: torch.complex32,
            torch.float32: torch.complex64,
            torch.float64: torch.complex128
        }[dtype]

        def _test_complex(sizes, signal_ndim, prepro_fn=lambda x: x):
            x = prepro_fn(torch.randn(*sizes, dtype=complex_dtype, device=device))
            dim = tuple(range(-signal_ndim, 0))
            for norm in ('ortho', None):
                res = torch.fft.fftn(x, dim=dim, norm=norm)
                rec = torch.fft.ifftn(res, dim=dim, norm=norm)
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='fft and ifft')
                res = torch.fft.ifftn(x, dim=dim, norm=norm)
                rec = torch.fft.fftn(res, dim=dim, norm=norm)
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='ifft and fft')

        def _test_real(sizes, signal_ndim, prepro_fn=lambda x: x):
            x = prepro_fn(torch.randn(*sizes, dtype=dtype, device=device))
            signal_numel = 1
            signal_sizes = x.size()[-signal_ndim:]
            dim = tuple(range(-signal_ndim, 0))
            for norm in (None, 'ortho'):
                res = torch.fft.rfftn(x, dim=dim, norm=norm)
                rec = torch.fft.irfftn(res, s=signal_sizes, dim=dim, norm=norm)
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='rfft and irfft')
                res = torch.fft.fftn(x, dim=dim, norm=norm)
                rec = torch.fft.ifftn(res, dim=dim, norm=norm)
                x_complex = torch.complex(x, torch.zeros_like(x))
                self.assertEqual(x_complex, rec, atol=1e-8, rtol=0, msg='fft and ifft (from real)')

        # contiguous case
        _test_real((100,), 1)
        _test_real((10, 1, 10, 100), 1)
        _test_real((100, 100), 2)
        _test_real((2, 2, 5, 80, 60), 2)
        _test_real((50, 40, 70), 3)
        _test_real((30, 1, 50, 25, 20), 3)

        _test_complex((100,), 1)
        _test_complex((100, 100), 1)
        _test_complex((100, 100), 2)
        _test_complex((1, 20, 80, 60), 2)
        _test_complex((50, 40, 70), 3)
        _test_complex((6, 5, 50, 25, 20), 3)

        # non-contiguous case
        _test_real((165,), 1, lambda x: x.narrow(0, 25, 100))  # input is not aligned to complex type
        _test_real((100, 100, 3), 1, lambda x: x[:, :, 0])
        _test_real((100, 100), 2, lambda x: x.t())
        _test_real((20, 100, 10, 10), 2, lambda x: x.view(20, 100, 100)[:, :60])
        _test_real((65, 80, 115), 3, lambda x: x[10:60, 13:53, 10:80])
        _test_real((30, 20, 50, 25), 3, lambda x: x.transpose(1, 2).transpose(2, 3))

        _test_complex((100,), 1, lambda x: x.expand(100, 100))
        _test_complex((20, 90, 110), 2, lambda x: x[:, 5:85].narrow(2, 5, 100))
        _test_complex((40, 60, 3, 80), 3, lambda x: x.transpose(2, 0).select(0, 2)[5:55, :, 10:])
        _test_complex((30, 55, 50, 22), 3, lambda x: x[:, 3:53, 15:40, 1:21])

    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.double)
    def test_fft_ifft_rfft_irfft(self, device, dtype):
        self._test_fft_ifft_rfft_irfft(device, dtype)

    @deviceCountAtLeast(1)
    @skipCUDAIfRocm
    @onlyCUDA
    @dtypes(torch.double)
    def test_cufft_plan_cache(self, devices, dtype):
        @contextmanager
        def plan_cache_max_size(device, n):
            if device is None:
                plan_cache = torch.backends.cuda.cufft_plan_cache
            else:
                plan_cache = torch.backends.cuda.cufft_plan_cache[device]
            original = plan_cache.max_size
            plan_cache.max_size = n
            yield
            plan_cache.max_size = original

        with plan_cache_max_size(devices[0], max(1, torch.backends.cuda.cufft_plan_cache.size - 10)):
            self._test_fft_ifft_rfft_irfft(devices[0], dtype)

        with plan_cache_max_size(devices[0], 0):
            self._test_fft_ifft_rfft_irfft(devices[0], dtype)

        torch.backends.cuda.cufft_plan_cache.clear()

        # check that stll works after clearing cache
        with plan_cache_max_size(devices[0], 10):
            self._test_fft_ifft_rfft_irfft(devices[0], dtype)

        with self.assertRaisesRegex(RuntimeError, r"must be non-negative"):
            torch.backends.cuda.cufft_plan_cache.max_size = -1

        with self.assertRaisesRegex(RuntimeError, r"read-only property"):
            torch.backends.cuda.cufft_plan_cache.size = -1

        with self.assertRaisesRegex(RuntimeError, r"but got device with index"):
            torch.backends.cuda.cufft_plan_cache[torch.cuda.device_count() + 10]

        # Multigpu tests
        if len(devices) > 1:
            # Test that different GPU has different cache
            x0 = torch.randn(2, 3, 3, device=devices[0])
            x1 = x0.to(devices[1])
            self.assertEqual(torch.fft.rfftn(x0, dim=(-2, -1)), torch.fft.rfftn(x1, dim=(-2, -1)))
            # If a plan is used across different devices, the following line (or
            # the assert above) would trigger illegal memory access. Other ways
            # to trigger the error include
            #   (1) setting CUDA_LAUNCH_BLOCKING=1 (pytorch/pytorch#19224) and
            #   (2) printing a device 1 tensor.
            x0.copy_(x1)

            # Test that un-indexed `torch.backends.cuda.cufft_plan_cache` uses current device
            with plan_cache_max_size(devices[0], 10):
                with plan_cache_max_size(devices[1], 11):
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                    self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                    with torch.cuda.device(devices[1]):
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(devices[0]):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0

                self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                with torch.cuda.device(devices[1]):
                    with plan_cache_max_size(None, 11):  # default is cuda:1
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(devices[0]):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1

    # passes on ROCm w/ python 2.7, fails w/ python 3.6
    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.double)
    def test_stft(self, device, dtype):
        if not TEST_LIBROSA:
            raise unittest.SkipTest('librosa not found')

        def librosa_stft(x, n_fft, hop_length, win_length, window, center):
            if window is None:
                window = np.ones(n_fft if win_length is None else win_length)
            else:
                window = window.cpu().numpy()
            input_1d = x.dim() == 1
            if input_1d:
                x = x.view(1, -1)
            result = []
            for xi in x:
                ri = librosa.stft(xi.cpu().numpy(), n_fft, hop_length, win_length, window, center=center)
                result.append(torch.from_numpy(np.stack([ri.real, ri.imag], -1)))
            result = torch.stack(result, 0)
            if input_1d:
                result = result[0]
            return result

        def _test(sizes, n_fft, hop_length=None, win_length=None, win_sizes=None,
                  center=True, expected_error=None):
            x = torch.randn(*sizes, dtype=dtype, device=device)
            if win_sizes is not None:
                window = torch.randn(*win_sizes, dtype=dtype, device=device)
            else:
                window = None
            if expected_error is None:
                result = x.stft(n_fft, hop_length, win_length, window,
                                center=center, return_complex=False)
                # NB: librosa defaults to np.complex64 output, no matter what
                # the input dtype
                ref_result = librosa_stft(x, n_fft, hop_length, win_length, window, center)
                self.assertEqual(result, ref_result, atol=7e-6, rtol=0, msg='stft comparison against librosa', exact_dtype=False)
                # With return_complex=True, the result is the same but viewed as complex instead of real
                result_complex = x.stft(n_fft, hop_length, win_length, window, center=center, return_complex=True)
                self.assertEqual(result_complex, torch.view_as_complex(result))
            else:
                self.assertRaises(expected_error,
                                  lambda: x.stft(n_fft, hop_length, win_length, window, center=center))

        for center in [True, False]:
            _test((10,), 7, center=center)
            _test((10, 4000), 1024, center=center)

            _test((10,), 7, 2, center=center)
            _test((10, 4000), 1024, 512, center=center)

            _test((10,), 7, 2, win_sizes=(7,), center=center)
            _test((10, 4000), 1024, 512, win_sizes=(1024,), center=center)

            # spectral oversample
            _test((10,), 7, 2, win_length=5, center=center)
            _test((10, 4000), 1024, 512, win_length=100, center=center)

        _test((10, 4, 2), 1, 1, expected_error=RuntimeError)
        _test((10,), 11, 1, center=False, expected_error=RuntimeError)
        _test((10,), -1, 1, expected_error=RuntimeError)
        _test((10,), 3, win_length=5, expected_error=RuntimeError)
        _test((10,), 5, 4, win_sizes=(11,), expected_error=RuntimeError)
        _test((10,), 5, 4, win_sizes=(1, 1), expected_error=RuntimeError)


    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double, torch.cdouble)
    def test_complex_stft_roundtrip(self, device, dtype):
        test_args = list(product(
            # input
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype),
             torch.randn(12, 60, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # center
            (True,),
            # pad_mode
            ("constant", "reflect", "circular"),
            # normalized
            (True, False),
            # onesided
            (True, False) if not dtype.is_complex else (False,),
        ))

        for args in test_args:
            x, n_fft, hop_length, center, pad_mode, normalized, onesided = args
            common_kwargs = {
                'n_fft': n_fft, 'hop_length': hop_length, 'center': center,
                'normalized': normalized, 'onesided': onesided,
            }

            # Functional interface
            x_stft = torch.stft(x, pad_mode=pad_mode, return_complex=True, **common_kwargs)
            x_roundtrip = torch.istft(x_stft, return_complex=dtype.is_complex,
                                      length=x.size(-1), **common_kwargs)
            self.assertEqual(x_roundtrip, x)

            # Tensor method interface
            x_stft = x.stft(pad_mode=pad_mode, return_complex=True, **common_kwargs)
            x_roundtrip = torch.istft(x_stft, return_complex=dtype.is_complex,
                                      length=x.size(-1), **common_kwargs)
            self.assertEqual(x_roundtrip, x)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double, torch.cdouble)
    def test_stft_roundtrip_complex_window(self, device, dtype):
        test_args = list(product(
            # input
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype),
             torch.randn(12, 60, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # pad_mode
            ("constant", "reflect", "replicate", "circular"),
            # normalized
            (True, False),
        ))
        for args in test_args:
            x, n_fft, hop_length, pad_mode, normalized = args
            window = torch.rand(n_fft, device=device, dtype=torch.cdouble)
            x_stft = torch.stft(
                x, n_fft=n_fft, hop_length=hop_length, window=window,
                center=True, pad_mode=pad_mode, normalized=normalized)
            self.assertEqual(x_stft.dtype, torch.cdouble)
            self.assertEqual(x_stft.size(-2), n_fft)  # Not onesided

            x_roundtrip = torch.istft(
                x_stft, n_fft=n_fft, hop_length=hop_length, window=window,
                center=True, normalized=normalized, length=x.size(-1),
                return_complex=True)
            self.assertEqual(x_stft.dtype, torch.cdouble)

            if not dtype.is_complex:
                self.assertEqual(x_roundtrip.imag, torch.zeros_like(x_roundtrip.imag),
                                 atol=1e-6, rtol=0)
                self.assertEqual(x_roundtrip.real, x)
            else:
                self.assertEqual(x_roundtrip, x)


    @skipCPUIfNoMkl
    @dtypes(torch.cdouble)
    def test_complex_stft_definition(self, device, dtype):
        test_args = list(product(
            # input
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (10, 15)
        ))

        for args in test_args:
            window = torch.randn(args[1], device=device, dtype=dtype)
            expected = _stft_reference(args[0], args[2], window)
            actual = torch.stft(*args, window=window, center=False)
            self.assertEqual(actual, expected)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.cdouble)
    def test_complex_stft_real_equiv(self, device, dtype):
        test_args = list(product(
            # input
            (torch.rand(600, device=device, dtype=dtype),
             torch.rand(807, device=device, dtype=dtype),
             torch.rand(14, 50, device=device, dtype=dtype),
             torch.rand(6, 51, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # win_length
            (None, 20),
            # center
            (False, True),
            # pad_mode
            ("constant", "reflect", "circular"),
            # normalized
            (True, False),
        ))

        for args in test_args:
            x, n_fft, hop_length, win_length, center, pad_mode, normalized = args
            expected = _complex_stft(x, n_fft, hop_length=hop_length,
                                     win_length=win_length, pad_mode=pad_mode,
                                     center=center, normalized=normalized)
            actual = torch.stft(x, n_fft, hop_length=hop_length,
                                win_length=win_length, pad_mode=pad_mode,
                                center=center, normalized=normalized)
            self.assertEqual(expected, actual)

    @skipCPUIfNoMkl
    @dtypes(torch.cdouble)
    def test_complex_istft_real_equiv(self, device, dtype):
        test_args = list(product(
            # input
            (torch.rand(40, 20, device=device, dtype=dtype),
             torch.rand(25, 1, device=device, dtype=dtype),
             torch.rand(4, 20, 10, device=device, dtype=dtype)),
            # hop_length
            (None, 10),
            # center
            (False, True),
            # normalized
            (True, False),
        ))

        for args in test_args:
            x, hop_length, center, normalized = args
            n_fft = x.size(-2)
            expected = _complex_istft(x, n_fft, hop_length=hop_length,
                                      center=center, normalized=normalized)
            actual = torch.istft(x, n_fft, hop_length=hop_length,
                                 center=center, normalized=normalized,
                                 return_complex=True)
            self.assertEqual(expected, actual)

    @skipCPUIfNoMkl
    def test_complex_stft_onesided(self, device):
        # stft of complex input cannot be onesided
        for x_dtype, window_dtype in product((torch.double, torch.cdouble), repeat=2):
            x = torch.rand(100, device=device, dtype=x_dtype)
            window = torch.rand(10, device=device, dtype=window_dtype)

            if x_dtype.is_complex or window_dtype.is_complex:
                with self.assertRaisesRegex(RuntimeError, 'complex'):
                    x.stft(10, window=window, pad_mode='constant', onesided=True)
            else:
                y = x.stft(10, window=window, pad_mode='constant', onesided=True,
                           return_complex=True)
                self.assertEqual(y.dtype, torch.cdouble)
                self.assertEqual(y.size(), (6, 51))

        x = torch.rand(100, device=device, dtype=torch.cdouble)
        with self.assertRaisesRegex(RuntimeError, 'complex'):
            x.stft(10, pad_mode='constant', onesided=True)

    # stft is currently warning that it requires return-complex while an upgrader is written
    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    def test_stft_requires_complex(self, device):
        x = torch.rand(100)
        y = x.stft(10, pad_mode='constant')
        # with self.assertRaisesRegex(RuntimeError, 'stft requires the return_complex parameter'):
        #     y = x.stft(10, pad_mode='constant')

    @skipCPUIfNoMkl
    def test_fft_input_modification(self, device):
        # FFT functions should not modify their input (gh-34551)

        signal = torch.ones((2, 2, 2), device=device)
        signal_copy = signal.clone()
        spectrum = torch.fft.fftn(signal, dim=(-2, -1))
        self.assertEqual(signal, signal_copy)

        spectrum_copy = spectrum.clone()
        _ = torch.fft.ifftn(spectrum, dim=(-2, -1))
        self.assertEqual(spectrum, spectrum_copy)

        half_spectrum = torch.fft.rfftn(signal, dim=(-2, -1))
        self.assertEqual(signal, signal_copy)

        half_spectrum_copy = half_spectrum.clone()
        _ = torch.fft.irfftn(half_spectrum_copy, s=(2, 2), dim=(-2, -1))
        self.assertEqual(half_spectrum, half_spectrum_copy)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_round_trip_simple_cases(self, device, dtype):
        """stft -> istft should recover the original signale"""
        def _test(input, n_fft, length):
            stft = torch.stft(input, n_fft=n_fft, return_complex=True)
            inverse = torch.istft(stft, n_fft=n_fft, length=length)
            self.assertEqual(input, inverse, exact_dtype=True)

        _test(torch.ones(4, dtype=dtype, device=device), 4, 4)
        _test(torch.zeros(4, dtype=dtype, device=device), 4, 4)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_round_trip_various_params(self, device, dtype):
        """stft -> istft should recover the original signale"""
        def _test_istft_is_inverse_of_stft(stft_kwargs):
            # generates a random sound signal for each tril and then does the stft/istft
            # operation to check whether we can reconstruct signal
            data_sizes = [(2, 20), (3, 15), (4, 10)]
            num_trials = 100
            istft_kwargs = stft_kwargs.copy()
            del istft_kwargs['pad_mode']
            for sizes in data_sizes:
                for i in range(num_trials):
                    original = torch.randn(*sizes, dtype=dtype, device=device)
                    stft = torch.stft(original, return_complex=True, **stft_kwargs)
                    inversed = torch.istft(stft, length=original.size(1), **istft_kwargs)

                    # trim the original for case when constructed signal is shorter than original
                    original = original[..., :inversed.size(-1)]
                    self.assertEqual(
                        inversed, original, msg='istft comparison against original',
                        atol=7e-6, rtol=0, exact_dtype=True)

        patterns = [
            # hann_window, centered, normalized, onesided
            {
                'n_fft': 12,
                'hop_length': 4,
                'win_length': 12,
                'window': torch.hann_window(12, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'reflect',
                'normalized': True,
                'onesided': True,
            },
            # hann_window, centered, not normalized, not onesided
            {
                'n_fft': 12,
                'hop_length': 2,
                'win_length': 8,
                'window': torch.hann_window(8, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'reflect',
                'normalized': False,
                'onesided': False,
            },
            # hamming_window, centered, normalized, not onesided
            {
                'n_fft': 15,
                'hop_length': 3,
                'win_length': 11,
                'window': torch.hamming_window(11, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'constant',
                'normalized': True,
                'onesided': False,
            },
            # hamming_window, not centered, not normalized, onesided
            # window same size as n_fft
            {
                'n_fft': 5,
                'hop_length': 2,
                'win_length': 5,
                'window': torch.hamming_window(5, dtype=dtype, device=device),
                'center': False,
                'pad_mode': 'constant',
                'normalized': False,
                'onesided': True,
            },
            # hamming_window, not centered, not normalized, not onesided
            # window same size as n_fft
            {
                'n_fft': 3,
                'hop_length': 2,
                'win_length': 3,
                'window': torch.hamming_window(3, dtype=dtype, device=device),
                'center': False,
                'pad_mode': 'reflect',
                'normalized': False,
                'onesided': False,
            },
        ]
        for i, pattern in enumerate(patterns):
            _test_istft_is_inverse_of_stft(pattern)

    @onlyOnCPUAndCUDA
    def test_istft_throws(self, device):
        """istft should throw exception for invalid parameters"""
        stft = torch.zeros((3, 5, 2), device=device)
        # the window is size 1 but it hops 20 so there is a gap which throw an error
        self.assertRaises(
            RuntimeError, torch.istft, stft, n_fft=4,
            hop_length=20, win_length=1, window=torch.ones(1))
        # A window of zeros does not meet NOLA
        invalid_window = torch.zeros(4, device=device)
        self.assertRaises(
            RuntimeError, torch.istft, stft, n_fft=4, win_length=4, window=invalid_window)
        # Input cannot be empty
        self.assertRaises(RuntimeError, torch.istft, torch.zeros((3, 0, 2)), 2)
        self.assertRaises(RuntimeError, torch.istft, torch.zeros((0, 3, 2)), 2)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_of_sine(self, device, dtype):
        def _test(amplitude, L, n):
            # stft of amplitude*sin(2*pi/L*n*x) with the hop length and window size equaling L
            x = torch.arange(2 * L + 1, device=device, dtype=dtype)
            original = amplitude * torch.sin(2 * math.pi / L * x * n)
            # stft = torch.stft(original, L, hop_length=L, win_length=L,
            #                   window=torch.ones(L), center=False, normalized=False)
            stft = torch.zeros((L // 2 + 1, 2, 2), device=device, dtype=dtype)
            stft_largest_val = (amplitude * L) / 2.0
            if n < stft.size(0):
                stft[n, :, 1] = -stft_largest_val

            if 0 <= L - n < stft.size(0):
                # symmetric about L // 2
                stft[L - n, :, 1] = stft_largest_val

            inverse = torch.istft(
                stft, L, hop_length=L, win_length=L,
                window=torch.ones(L, device=device, dtype=dtype), center=False, normalized=False)
            # There is a larger error due to the scaling of amplitude
            original = original[..., :inverse.size(-1)]
            self.assertEqual(inverse, original, atol=1e-3, rtol=0)

        _test(amplitude=123, L=5, n=1)
        _test(amplitude=150, L=5, n=2)
        _test(amplitude=111, L=5, n=3)
        _test(amplitude=160, L=7, n=4)
        _test(amplitude=145, L=8, n=5)
        _test(amplitude=80, L=9, n=6)
        _test(amplitude=99, L=10, n=7)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_linearity(self, device, dtype):
        num_trials = 100

        def _test(data_size, kwargs):
            for i in range(num_trials):
                tensor1 = torch.randn(data_size, device=device, dtype=dtype)
                tensor2 = torch.randn(data_size, device=device, dtype=dtype)
                a, b = torch.rand(2, dtype=dtype, device=device)
                # Also compare method vs. functional call signature
                istft1 = tensor1.istft(**kwargs)
                istft2 = tensor2.istft(**kwargs)
                istft = a * istft1 + b * istft2
                estimate = torch.istft(a * tensor1 + b * tensor2, **kwargs)
                self.assertEqual(istft, estimate, atol=1e-5, rtol=0)
        patterns = [
            # hann_window, centered, normalized, onesided
            (
                (2, 7, 7, 2),
                {
                    'n_fft': 12,
                    'window': torch.hann_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': True,
                    'onesided': True,
                },
            ),
            # hann_window, centered, not normalized, not onesided
            (
                (2, 12, 7, 2),
                {
                    'n_fft': 12,
                    'window': torch.hann_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': False,
                    'onesided': False,
                },
            ),
            # hamming_window, centered, normalized, not onesided
            (
                (2, 12, 7, 2),
                {
                    'n_fft': 12,
                    'window': torch.hamming_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': True,
                    'onesided': False,
                },
            ),
            # hamming_window, not centered, not normalized, onesided
            (
                (2, 7, 3, 2),
                {
                    'n_fft': 12,
                    'window': torch.hamming_window(12, device=device, dtype=dtype),
                    'center': False,
                    'normalized': False,
                    'onesided': True,
                },
            )
        ]
        for data_size, kwargs in patterns:
            _test(data_size, kwargs)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    def test_batch_istft(self, device):
        original = torch.tensor([
            [[4., 0.], [4., 0.], [4., 0.], [4., 0.], [4., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        ], device=device)

        single = original.repeat(1, 1, 1, 1)
        multi = original.repeat(4, 1, 1, 1)

        i_original = torch.istft(original, n_fft=4, length=4)
        i_single = torch.istft(single, n_fft=4, length=4)
        i_multi = torch.istft(multi, n_fft=4, length=4)

        self.assertEqual(i_original.repeat(1, 1), i_single, atol=1e-6, rtol=0, exact_dtype=True)
        self.assertEqual(i_original.repeat(4, 1), i_multi, atol=1e-6, rtol=0, exact_dtype=True)

    @onlyCUDA
    @skipIf(not TEST_MKL, "Test requires MKL")
    def test_stft_window_device(self, device):
        # Test the (i)stft window must be on the same device as the input
        x = torch.randn(1000, dtype=torch.complex64)
        window = torch.randn(100, dtype=torch.complex64)

        with self.assertRaisesRegex(RuntimeError, "stft input and window must be on the same device"):
            torch.stft(x, n_fft=100, window=window.to(device))

        with self.assertRaisesRegex(RuntimeError, "stft input and window must be on the same device"):
            torch.stft(x.to(device), n_fft=100, window=window)

        X = torch.stft(x, n_fft=100, window=window)

        with self.assertRaisesRegex(RuntimeError, "istft input and window must be on the same device"):
            torch.istft(X, n_fft=100, window=window.to(device))

        with self.assertRaisesRegex(RuntimeError, "istft input and window must be on the same device"):
            torch.istft(x.to(device), n_fft=100, window=window)


instantiate_device_type_tests(TestFFT, globals())

if __name__ == '__main__':
    run_tests()
