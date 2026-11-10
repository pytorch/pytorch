# Owner(s): ["module: type promotion"]

import math

import numpy as np

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
    torch.float8_e5m3fnu,
)
# These only represent magnitudes: the conversions round |x|.
UNSIGNED_DTYPES = (torch.float8_e8m0fnu, torch.float8_e5m3fnu)
# Powers of two only: ties round up, as there are no mantissa bits to be even.
NO_MANTISSA_DTYPES = (torch.float8_e8m0fnu,)
NARROWING = {
    torch.float64: (torch.float32, torch.float16, torch.bfloat16, *FLOAT8_DTYPES),
    torch.float32: (torch.float16, torch.bfloat16, *FLOAT8_DTYPES),
    torch.float16: (torch.bfloat16, *FLOAT8_DTYPES),
    torch.bfloat16: (torch.float16, *FLOAT8_DTYPES),
}
PAIRS = [(src, dst) for src, dsts in NARROWING.items() for dst in dsts]


def _pair_name(pair):
    src, dst = pair
    return f"{str(src).removeprefix('torch.')}_to_{str(dst).removeprefix('torch.')}"


def _finite_magnitudes(dtype):
    """All finite non-negative values of dtype, in encoding order."""
    bits = torch.finfo(dtype).bits
    signed = dtype not in UNSIGNED_DTYPES
    n = 1 << (bits - 1) if signed else 1 << bits
    int_dtype = torch.int16 if bits == 16 else torch.uint8
    values = torch.arange(n, dtype=torch.int32).to(int_dtype).view(dtype).double()
    finite = values.isfinite()
    num_finite = int(finite.sum())
    if not finite[:num_finite].all():
        raise AssertionError(f"finite encodings of {dtype} must be a prefix")
    mags = values[:num_finite]
    if not (mags[1:] > mags[:-1]).all():
        raise AssertionError(f"magnitudes of {dtype} must increase")
    return mags


def _overflow_value(dtype):
    """The default conversion's result past the largest finite magnitude."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m3fnu):
        return torch.finfo(dtype).max
    if dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz, torch.float8_e8m0fnu):
        return math.nan
    return math.inf


def _virtual_next(mags, dst):
    """The value after the largest magnitude mags[-1], were the exponent range
    wider."""
    if dst in NO_MANTISSA_DTYPES:
        return mags[-1] * 2
    return 2 * mags[-1] - mags[-2]


def _expected_narrow(x, dst):
    """Reference rounding to nearest of the float64 values x (exact source
    values) to dst, with ties to even (or up, for dtypes without mantissa),
    computed from a table of all of dst's finite values."""
    mags = _finite_magnitudes(dst)
    n = len(mags)
    signed = dst not in UNSIGNED_DTYPES
    has_neg_zero = signed and dst not in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    no_mantissa = dst in NO_MANTISSA_DTYPES
    overflow_value = _overflow_value(dst)
    virtual_next = _virtual_next(mags, dst)

    neg = torch.signbit(x) & signed
    a = torch.where(x.isfinite(), x.abs(), torch.zeros_like(x))
    down = torch.searchsorted(mags, a, right=True) - 1
    below = down < 0  # only for dtypes without zero
    down = down.clamp(min=0)
    up = torch.where((mags[down] == a) | below, down, down + 1)
    up_value = torch.where(up < n, mags[up.clamp(max=n - 1)], virtual_next)
    mid = (mags[down] + up_value) / 2
    tie = up if no_mantissa else torch.where(down % 2 == 0, down, up)
    r = torch.where(a < mid, down, torch.where(a > mid, up, tie))

    value = mags[r.clamp(max=n - 1)]
    value = torch.where(r >= n, torch.full_like(value, overflow_value), value)
    value = torch.where(x.isinf(), torch.full_like(value, overflow_value), value)
    value = torch.where(neg, -value, value)
    if not has_neg_zero:
        value = torch.where(value == 0, torch.zeros_like(value), value)
    return torch.where(x.isnan(), torch.full_like(value, math.nan), value)


def _expected(x, dst):
    if dst is torch.float32:
        with np.errstate(over="ignore", invalid="ignore"):
            return torch.from_numpy(x.numpy().astype(np.float32)).double()
    return _expected_narrow(x, dst)


def _boundary_inputs(src, dst):
    """Values of dst, midpoints between consecutive values, and their nearest
    src neighbors, with both signs, plus special values."""
    if dst is torch.float32:
        # Too many values to enumerate: use random float32 values instead.
        mags = (
            torch.randint(0, 0x7F800000, (1 << 16,), dtype=torch.int32)
            .view(torch.float32)
            .double()
        )
        mags = mags.sort().values
        top = torch.tensor([torch.finfo(dst).max, 2.0**128], dtype=torch.float64)
        # Midpoints between the random values and the next float32 values.
        next_mags = torch.nextafter(mags.float(), torch.tensor(math.inf)).double()
        mags = torch.cat([mags, next_mags]).sort().values
    else:
        mags = _finite_magnitudes(dst)
        top = torch.stack([mags[-1], _virtual_next(mags, dst)])
    # Includes the midpoint between the largest value and the next one, were
    # the exponent range wider.
    centers = torch.cat(
        [mags, (mags[1:] + mags[:-1]) / 2, top.mean(0, keepdim=True), mags[-1:] * 4]
    )
    centers = torch.cat([centers, -centers]).to(src)
    inputs = [centers]
    up = down = centers
    for _ in range(3):
        up = torch.nextafter(up, torch.full_like(up, math.inf))
        down = torch.nextafter(down, torch.full_like(down, -math.inf))
        inputs += [up, down]
    finfo = torch.finfo(src)
    specials = [
        0.0,
        -0.0,
        math.inf,
        -math.inf,
        math.nan,
        finfo.max,
        -finfo.max,
        finfo.tiny,
        finfo.smallest_normal / 3,
        -finfo.tiny / 3,
    ]
    inputs.append(torch.tensor(specials, dtype=src))
    return torch.cat(inputs)


def _random_inputs(src, n=1 << 18):
    """Random bit patterns, biased toward the range of narrow formats."""
    if src is torch.float32:
        return torch.randint(-(2**31), 2**31 - 1, (n,), dtype=torch.int32).view(
            torch.float32
        )
    exponents = torch.randint(1023 - 160, 1023 + 160, (n,), dtype=torch.int64)
    mantissas = torch.randint(0, 2**52, (n,), dtype=torch.int64)
    signs = torch.randint(0, 2, (n,), dtype=torch.int64) << 63
    return (signs | (exponents << 52) | mantissas).view(torch.float64)


def _all_inputs(src, dst):
    if src in (torch.float16, torch.bfloat16):
        return (
            torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(src)
        )
    return torch.cat([_boundary_inputs(src, dst), _random_inputs(src)])


class TestRoundedConversion(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def assertSameValues(self, actual, expected, inputs):
        actual = actual.cpu().double()
        expected = expected.cpu()
        nan = expected.isnan()
        self.assertEqual(actual.isnan(), nan)
        bad = ~nan & (
            (actual != expected) | (torch.signbit(actual) != torch.signbit(expected))
        )
        if bad.any():
            i = bad.nonzero()[0].item()
            self.fail(
                f"{int(bad.sum())} mismatches, e.g. input {inputs[i].item()!r} -> "
                f"{actual[i].item()!r}, expected {expected[i].item()!r}"
            )

    @parametrize("pair", PAIRS, name_fn=_pair_name)
    def test_default_matches_reference(self, device, pair):
        src, dst = pair
        x = _all_inputs(src, dst)
        expected = _expected(x.double(), dst)
        self.assertSameValues(x.to(device).to(dst), expected, x)

    @dtypes(torch.float16, torch.bfloat16)
    def test_double_rounding(self, device, dtype):
        # Just above the midpoint between 1 and the next value of dtype, but
        # rounding to float32 first would land exactly on the midpoint.
        # (bfloat16 catches direct static_casts from double on every platform:
        # float16 has a native conversion on aarch64.)
        eps = torch.finfo(dtype).eps
        v = 1 + eps / 2 + 2**-40
        x = torch.tensor([v], dtype=torch.float64, device=device)
        self.assertEqual(x.to(dtype).item(), 1 + eps)
        self.assertEqual(x.to(torch.float32).to(dtype).item(), 1.0)
        # Other conversions: from a Scalar (fill_ special-cases one element),
        # and of an operator's output, where chebyshev_polynomial_t
        # (T_1(x) = x) is jiterated on CUDA.
        for n in (1, 2):
            y = torch.full((n,), v, dtype=dtype, device=device)
            self.assertEqual(y.tolist(), [1 + eps] * n)
        if dtype is torch.bfloat16:
            # All fill paths agree on integer Scalars (which still round
            # through float) and complex Scalars.
            for value in (2**24 + 2**16 + 1, complex(v, 0)):
                ys = [
                    torch.full((n,), value, dtype=dtype, device=device) for n in (1, 2)
                ]
                self.assertEqual(ys[0][0], ys[1][0])
        out = torch.zeros(1, dtype=dtype, device=device)
        self.assertEqual(torch.add(x, 0, out=out).item(), 1 + eps)
        out.zero_()
        torch.special.chebyshev_polynomial_t(x, 1, out=out)
        self.assertEqual(out.item(), 1 + eps)

    def test_double_rounding_complex32(self, device):
        v = 1 + 2**-11 + 2**-40
        expected = complex(1 + 2**-10, -(1 + 2**-10))
        z = torch.tensor([complex(v, -v)], dtype=torch.complex128, device=device)
        self.assertEqual(z.to(torch.complex32).to(torch.complex128).item(), expected)
        x = torch.tensor([v], dtype=torch.float64, device=device)
        w = x.to(torch.complex32).to(torch.complex128)
        self.assertEqual(w.item(), complex(1 + 2**-10, 0))
        w = torch.tensor([complex(v, -v)], dtype=torch.complex32, device=device)
        self.assertEqual(w.to(torch.complex128).item(), expected)
        for n in (1, 2):
            w = torch.full((n,), complex(v, -v), dtype=torch.complex32, device=device)
            self.assertEqual(w.to(torch.complex128).tolist(), [expected] * n)
            # Every complex32 fill checks for overflow, like float16 fills of
            # two or more elements and CUDA float16 fills.
            with self.assertRaisesRegex(RuntimeError, "without overflow"):
                torch.full((n,), complex(1e5, 0), dtype=torch.complex32, device=device)


instantiate_device_type_tests(TestRoundedConversion, globals())

if __name__ == "__main__":
    run_tests()
