# Owner(s): ["module: type promotion"]

import math

import numpy as np

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    slowTest,
    TestCase,
)


MODES = ("to_nearest_even", "toward_zero", "toward_positive", "toward_negative")

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
    def test_matches_reference(self, device, pair):
        src, dst = pair
        x = _all_inputs(src, dst)
        x_dev = x.to(device)
        for mode in MODES:
            with self.subTest(mode=mode):
                actual = x_dev.to(dst, round=mode)
                self.assertEqual(actual.dtype, dst)
                self.assertEqual(actual.device, x_dev.device)
                self.assertSameValues(actual, _expected(x.double(), dst, mode), x)

    @parametrize("pair", PAIRS, name_fn=_pair_name)
    def test_default_matches_reference(self, device, pair):
        src, dst = pair
        x = _all_inputs(src, dst)
        expected = _expected(x.double(), dst, "to_nearest_even")
        self.assertSameValues(x.to(device).to(dst), expected, x)

    def test_e8m0(self, device):
        x = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.5,
                1.25,
                0.75,
                3.0,
                2**-126,
                1.5 * 2**-127,
                2**-127,
                2**-149,
                2**127,
                1.5 * 2**127,
                torch.finfo(torch.float32).max,
                math.inf,
                -math.inf,
                math.nan,
            ],
            device=device,
        )

        def bits(mode):
            return x.to(torch.float8_e8m0fnu, round=mode).view(torch.uint8).tolist()

        # These match PTX cvt.rz/rp.ue8m0x2.f32 (non-satfinite).
        self.assertEqual(
            bits("toward_zero"),
            [
                0x00,
                0x00,
                0x7F,
                0x7F,
                0x7F,
                0x7E,
                0x80,
                0x01,
                0x00,
                0x00,
                0x00,
                0xFE,
                0xFE,
                0xFE,
                0xFF,
                0xFF,
                0xFF,
            ],
        )
        self.assertEqual(bits("toward_negative"), bits("toward_zero"))
        self.assertEqual(
            bits("toward_positive"),
            [
                0x00,
                0x00,
                0x7F,
                0x80,
                0x80,
                0x7F,
                0x81,
                0x01,
                0x01,
                0x00,
                0x00,
                0xFE,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
            ],
        )
        self.assertEqual(
            bits("to_nearest_even"),
            [
                0x00,
                0x00,
                0x7F,
                0x80,
                0x7F,
                0x7F,
                0x81,
                0x01,
                0x01,
                0x00,
                0x00,
                0xFE,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
            ],
        )

    def test_overflow(self, device):
        x = torch.tensor([1e6, -1e6, math.inf, -math.inf], device=device)
        cases = {
            # dtype: (toward_zero, toward_positive, toward_negative)
            torch.float16: (
                [65504, -65504, math.inf, -math.inf],
                [math.inf, -65504, math.inf, -math.inf],
                [65504, -math.inf, math.inf, -math.inf],
            ),
            torch.float8_e5m2: (
                [57344, -57344, math.inf, -math.inf],
                [math.inf, -57344, math.inf, -math.inf],
                [57344, -math.inf, math.inf, -math.inf],
            ),
            torch.float8_e4m3fn: (
                [448, -448, 448, -448],
                [448, -448, 448, -448],
                [448, -448, 448, -448],
            ),
            torch.float8_e4m3fnuz: (
                [240, -240, math.nan, math.nan],
                [math.nan, -240, math.nan, math.nan],
                [240, math.nan, math.nan, math.nan],
            ),
        }
        for dtype, expected in cases.items():
            for mode, values in zip(MODES[1:], expected):
                with self.subTest(dtype=dtype, mode=mode):
                    actual = x.to(dtype, round=mode).double().cpu()
                    self.assertEqual(
                        actual,
                        torch.tensor(values, dtype=torch.float64),
                        equal_nan=True,
                    )

    def test_double_rounding(self, device):
        # Just above the midpoint between 1 and the next float16, but rounding
        # to float32 first would land exactly on the midpoint.
        x = torch.tensor([1 + 2**-11 + 2**-40], dtype=torch.float64, device=device)
        self.assertEqual(x.to(torch.float16).item(), 1 + 2**-10)
        self.assertEqual(
            x.to(torch.float16, round="to_nearest_even").item(), 1 + 2**-10
        )
        self.assertEqual(x.to(torch.float32).to(torch.float16).item(), 1.0)

    def test_signatures(self, device):
        x = torch.randn(4, 5, device=device)
        expected = torch._to_dtype_rounded(x, torch.bfloat16, "toward_positive")
        other = torch.empty((), dtype=torch.bfloat16, device=device)
        for y in (
            x.to(torch.bfloat16, round="toward_positive"),
            x.to(dtype=torch.bfloat16, round="toward_positive"),
            x.to(torch.bfloat16, False, False, round="toward_positive"),
            x.to(device, torch.bfloat16, round="toward_positive"),
            x.to(other, round="toward_positive"),
            x.to(torch.bfloat16, copy=True, round="toward_positive"),
        ):
            self.assertEqual(y, expected, atol=0, rtol=0)
        out = torch.empty_like(expected)
        torch._to_dtype_rounded(x, torch.bfloat16, "toward_positive", out=out)
        self.assertEqual(out, expected, atol=0, rtol=0)

    def test_memory_format(self, device):
        x = torch.randn(2, 3, 4, 5, device=device).to(memory_format=torch.channels_last)
        y = x.to(torch.float16, round="toward_zero")
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))
        z = x.to(
            torch.float16, round="toward_zero", memory_format=torch.contiguous_format
        )
        self.assertTrue(z.is_contiguous())
        self.assertEqual(y, z, atol=0, rtol=0)
        t = torch.randn(5, 6, device=device).t()
        self.assertEqual(
            t.to(torch.float16, round="toward_zero"),
            t.contiguous().to(torch.float16, round="toward_zero"),
            atol=0,
            rtol=0,
        )

        # The layout change happens in the conversion itself, not in a copy.
        from torch.utils._python_dispatch import TorchDispatchMode

        class RecordOps(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.ops = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.ops.append(func)
                return func(*args, **(kwargs or {}))

        with RecordOps() as recorder:
            w = x.to(
                torch.float16,
                round="toward_zero",
                memory_format=torch.contiguous_format,
            )
        self.assertEqual(recorder.ops, [torch.ops.aten._to_dtype_rounded.default])
        self.assertTrue(w.is_contiguous())
        self.assertEqual(w, z, atol=0, rtol=0)

    def test_errors(self, device):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(ValueError, "round must be one of"):
            x.to(torch.float16, round="nearest")
        for src, dst in [
            (torch.float32, torch.float32),
            (torch.float16, torch.float32),
            (torch.float32, torch.float64),
            (torch.int32, torch.float16),
            (torch.float32, torch.int8),
            (torch.float32, torch.bool),
            (torch.complex64, torch.complex32),
            (torch.float32, torch.float4_e2m1fn_x2),
            (torch.float8_e4m3fn, torch.float8_e5m2),
        ]:
            with (
                self.subTest(src=src, dst=dst),
                self.assertRaisesRegex(RuntimeError, "round is only supported"),
            ):
                x.to(src).to(dst, round="toward_zero")
        with self.assertRaisesRegex(RuntimeError, "round is only supported"):
            x.to(device, round="toward_zero")
        with self.assertRaisesRegex(RuntimeError, "round is only supported"):
            torch._to_dtype_rounded(x, torch.float32, "toward_zero")

    def test_autograd(self, device):
        x = torch.randn(5, dtype=torch.float64, device=device, requires_grad=True)
        y = x.to(torch.float32, round="toward_positive")
        self.assertTrue(y.requires_grad)
        y.backward(torch.arange(5.0, device=device))
        self.assertEqual(x.grad, torch.arange(5.0, dtype=torch.float64, device=device))

        tangent = torch.randn(5, dtype=torch.float64, device=device)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(x.detach(), tangent)
            out = dual.to(torch.float32, round="toward_positive")
            _, out_tangent = torch.autograd.forward_ad.unpack_dual(out)
        self.assertEqual(
            out_tangent,
            tangent.to(torch.float32, round="toward_positive"),
            atol=0,
            rtol=0,
        )

    def test_meta_and_fake(self, device):
        x = torch.empty(4, 3, device="meta").t()
        y = x.to(torch.float8_e8m0fnu, round="toward_positive")
        self.assertEqual(y.dtype, torch.float8_e8m0fnu)
        self.assertEqual(y.stride(), x.stride())
        with self.assertRaisesRegex(RuntimeError, "round is only supported"):
            x.to(torch.float64, round="toward_positive")

        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            z = torch.empty(4, 3, device=device).to(torch.bfloat16, round="toward_zero")
        self.assertEqual(z.dtype, torch.bfloat16)
        self.assertEqual(z.device.type, torch.device(device).type)

    def test_vmap(self, device):
        x = torch.randn(3, 4, device=device)
        actual = torch.vmap(lambda t: t.to(torch.float16, round="toward_negative"))(x)
        self.assertEqual(
            actual, x.to(torch.float16, round="toward_negative"), atol=0, rtol=0
        )

    def test_compile(self, device):
        def fn(x):
            return (x * 3).to(torch.bfloat16, round="toward_positive") + 1

        x = torch.randn(64, device=device)
        self.assertEqual(torch.compile(fn, fullgraph=True)(x), fn(x), atol=0, rtol=0)

    def test_export(self, device):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float16, round="toward_zero")

        x = torch.randn(4, device=device)
        ep = torch.export.export(M(), (x,)).run_decompositions()
        targets = [n.target for n in ep.graph.nodes if n.op == "call_function"]
        self.assertIn(torch.ops.aten._to_dtype_rounded.default, targets)
        self.assertEqual(ep.module()(x), M()(x), atol=0, rtol=0)

    def test_torchscript(self, device):
        def fn(x):
            return x.to(torch.float16, round="toward_zero")

        x = torch.randn(4, device=device)
        self.assertEqual(torch.jit.script(fn)(x), fn(x), atol=0, rtol=0)

    @onlyAccelerator
    @slowTest
    def test_exhaustive_float32(self, device):
        # Every float32 input, comparing the accelerator (which uses hardware
        # instructions where available) with the CPU's software conversion.
        chunk = 1 << 28
        for start in range(0, 1 << 32, chunk):
            x = torch.arange(start, start + chunk, dtype=torch.int64).to(torch.int32)
            x = x.view(torch.float32)
            x_dev = x.to(device)
            for dst in (
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
                torch.float8_e8m0fnu,
            ):
                for mode in MODES:
                    actual = x_dev.to(dst, round=mode).cpu().float()
                    expected = x.to(dst, round=mode).float()
                    same = (actual == expected) & (
                        torch.signbit(actual) == torch.signbit(expected)
                    )
                    same |= actual.isnan() & expected.isnan()
                    self.assertTrue(same.all(), f"{dst} {mode} from 0x{start:08x}")

    @onlyAccelerator
    def test_cross_device(self, device):
        x = torch.randn(1000, dtype=torch.float64)
        expected = x.to(torch.float32, round="toward_positive")
        self.assertEqual(
            x.to(device, torch.float32, round="toward_positive").cpu(),
            expected,
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            x.to(device).to("cpu", torch.float32, round="toward_positive"),
            expected,
            atol=0,
            rtol=0,
        )


instantiate_device_type_tests(TestRoundedConversion, globals())


class TestRoundedConversionPython(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_refs(self):
        x = torch.randn(10, dtype=torch.float64)
        actual = torch._refs.to(x, torch.float16, round="toward_negative")
        self.assertEqual(
            actual, x.to(torch.float16, round="toward_negative"), atol=0, rtol=0
        )

    def test_packed_sequence(self):
        from torch.nn.utils.rnn import pack_padded_sequence

        data = torch.randn(3, 2, 4, dtype=torch.float64)
        packed = pack_padded_sequence(data, torch.tensor([3, 2]), enforce_sorted=False)
        converted = packed.to(torch.float16, round="toward_positive")
        self.assertEqual(
            converted.data,
            packed.data.to(torch.float16, round="toward_positive"),
            atol=0,
            rtol=0,
        )
        self.assertEqual(converted.sorted_indices, packed.sorted_indices)

    def test_module_to_rejects_round(self):
        with self.assertRaisesRegex(RuntimeError, "does not accept round argument"):
            torch.nn.Linear(2, 2).to(torch.float16, round="toward_zero")


instantiate_parametrized_tests(TestRoundedConversionPython)

if __name__ == "__main__":
    run_tests()
