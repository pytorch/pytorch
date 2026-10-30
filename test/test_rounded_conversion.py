# Owner(s): ["module: type promotion"]

import math

import numpy as np

import torch
from torch.testing._internal.common_device_type import (
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
)
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
    signed = dtype is not torch.float8_e8m0fnu
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
    if dtype is torch.float8_e4m3fn:
        return torch.finfo(dtype).max
    if dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz, torch.float8_e8m0fnu):
        return math.nan
    return math.inf


def _expected_narrow(x, dst, mode):
    """Reference rounding of the float64 values x (exact source values) to dst,
    computed from a table of all of dst's finite values."""
    mags = _finite_magnitudes(dst)
    n = len(mags)
    signed = dst is not torch.float8_e8m0fnu
    has_neg_zero = dst not in (
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
    )
    no_mantissa = dst is torch.float8_e8m0fnu
    overflow_value = _overflow_value(dst)
    virtual_next = mags[-1] * 2 if no_mantissa else 2 * mags[-1] - mags[-2]

    neg = torch.signbit(x) & signed
    a = torch.where(x.isfinite(), x.abs(), torch.zeros_like(x))
    down = torch.searchsorted(mags, a, right=True) - 1
    below = down < 0  # only for dtypes without zero
    down = down.clamp(min=0)
    up = torch.where((mags[down] == a) | below, down, down + 1)
    up_value = torch.where(up < n, mags[up.clamp(max=n - 1)], virtual_next)

    if mode == "to_nearest_even":
        to_zero = torch.zeros_like(neg)
        mid = (mags[down] + up_value) / 2
        tie = up if no_mantissa else torch.where(down % 2 == 0, down, up)
        r = torch.where(a < mid, down, torch.where(a > mid, up, tie))
    else:
        to_zero = {
            "toward_zero": torch.ones_like(neg),
            "toward_positive": neg,
            "toward_negative": ~neg,
        }[mode]
        r = torch.where(to_zero, down, up)

    value = mags[r.clamp(max=n - 1)]
    value = torch.where(
        r >= n,
        torch.where(to_zero, mags[-1], torch.full_like(value, overflow_value)),
        value,
    )
    value = torch.where(x.isinf(), torch.full_like(value, overflow_value), value)
    value = torch.where(neg, -value, value)
    if not has_neg_zero:
        value = torch.where(value == 0, torch.zeros_like(value), value)
    return torch.where(x.isnan(), torch.full_like(value, math.nan), value)


def _expected_float32(x, mode):
    """Reference rounding of float64 values to float32."""
    x = x.numpy()
    with np.errstate(over="ignore", invalid="ignore"):
        rn = x.astype(np.float32)
    if mode == "to_nearest_even":
        return torch.from_numpy(rn).double()
    neg = np.signbit(x)
    finite_max = np.float32(np.finfo(np.float32).max)
    down = np.where(
        np.isinf(rn) & np.isfinite(x), np.copysign(finite_max, x).astype(np.float32), rn
    )
    down = np.where(
        np.abs(down.astype(np.float64)) > np.abs(x),
        np.nextafter(down, np.float32(0)),
        down,
    )
    toward_inf = np.where(neg, np.float32(-np.inf), np.float32(np.inf))
    up = np.where(
        np.abs(down.astype(np.float64)) < np.abs(x),
        np.nextafter(down, toward_inf),
        down,
    )
    to_zero = {
        "toward_zero": np.ones_like(neg),
        "toward_positive": neg,
        "toward_negative": ~neg,
    }[mode]
    return torch.from_numpy(np.where(to_zero, down, up)).double()


def _expected(x, dst, mode):
    if dst is torch.float32:
        return _expected_float32(x, mode)
    return _expected_narrow(x, dst, mode)


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
    else:
        mags = _finite_magnitudes(dst)
    centers = torch.cat(
        [mags, (mags[1:] + mags[:-1]) / 2, mags[-1:] * 1.5, mags[-1:] * 4]
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
        expected = _expected(x.double(), dst, "to_nearest_even")
        self.assertSameValues(x.to(device).to(dst), expected, x)

    def test_double_rounding(self, device):
        # Just above the midpoint between 1 and the next float16, but rounding
        # to float32 first would land exactly on the midpoint.
        x = torch.tensor([1 + 2**-11 + 2**-40], dtype=torch.float64, device=device)
        self.assertEqual(x.to(torch.float16).item(), 1 + 2**-10)
        self.assertEqual(x.to(torch.float32).to(torch.float16).item(), 1.0)


instantiate_device_type_tests(TestRoundedConversion, globals())

if __name__ == "__main__":
    run_tests()
