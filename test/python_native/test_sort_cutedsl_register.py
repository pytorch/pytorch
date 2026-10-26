# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the CuTeDSL register sort of rows of at most 2048. It
# replaces the stock in-place leaves and must match them bit-exactly, including
# their order of special values: rows of at most 128 (comparison leaves) tie
# all NaNs above +inf, longer rows (radix leaf) put negative NaN first; both
# tie -0.0 with +0.0. The kernel always yields the stable order, so every case
# checks against the stable sort.

import itertools
import unittest

import torch
from torch._native import registry
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    slowTest,
    subtest,
    TestCase,
)


_KEY_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.float32,
    torch.int64,
    torch.float64,
)
_INDEX_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.uint16,
    torch.int16,
    torch.uint32,
    torch.int32,
)
# Each side of every rung (32 .. 2048) and leaf boundary (32, 128), plus a
# ragged row.
_NSORTS = (
    *(1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257),
    *(511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048),
)
_ROWS = (1, 7, 1000)
_BOUNDARY_NSORTS = (1, 32, 33, 128, 129, 2047, 2048)
# One key per code path for the length sweep (each spec is a JIT compile): two
# 1-byte keys per register, a 16-bit key and a 32-bit key packed with the
# index, and 64-bit keys beside it. Every dtype runs in test_index_dtypes.
_NSORT_KEY_DTYPES = (torch.uint8, torch.bfloat16, torch.float32, torch.int64)


def _slow_unless(fast, *axes):
    """Every combination of ``axes``; those ``fast`` rejects are @slowTest (each
    spec is a JIT compile, and the fast set is all CI runs)."""
    cases = [c[0] if len(axes) == 1 else c for c in itertools.product(*axes)]
    fast_ = fast if len(axes) == 1 else lambda c: fast(*c)
    return [c if fast_(c) else subtest(c, decorators=[slowTest]) for c in cases]


# (exponent, mantissa) bits.
_FLOAT_LAYOUT = {
    torch.float16: (5, 10),
    torch.bfloat16: (8, 7),
    torch.float32: (8, 23),
    torch.float64: (11, 52),
}
_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def _arch() -> int:
    from torch._native.ops.sort.cutedsl_impl import _arch

    return _arch(torch.cuda.current_device())


def _float_specials(dtype: torch.dtype) -> list[int]:
    """Bit patterns of zeros, infinities, NaN payloads (quiet, signalling,
    all-ones mantissa, both signs) and subnormals."""
    exponent, mantissa = _FLOAT_LAYOUT[dtype]
    inf = ((1 << exponent) - 1) << mantissa
    ones = (1 << mantissa) - 1
    positive = [0, inf, inf | 1 << (mantissa - 1), inf | 1, inf | ones, 1, ones]
    sign = 1 << (exponent + mantissa)
    return positive + [sign | bits for bits in positive]


def _make_keys(dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    # Few distinct values give many ties; a third of the keys are specials.
    if dtype.is_floating_point:
        x = torch.randint(-8, 8, shape, device="cuda").to(dtype) / 4
        bits = 8 * dtype.itemsize
        specials = [b - (b >> (bits - 1) << bits) for b in _float_specials(dtype)]
        view_dtype = _INT_VIEW[dtype.itemsize]
        pool = torch.tensor(specials, dtype=view_dtype, device="cuda")
    else:
        info = torch.iinfo(dtype)
        lo, hi = max(info.min, -100), min(info.max, 100)
        x = torch.randint(lo, hi + 1, shape, device="cuda", dtype=dtype)
        view_dtype = dtype
        extremes = (info.min, info.min + 1, info.max - 1, info.max)
        pool = torch.tensor(extremes, dtype=dtype, device="cuda")
    # Write the specials through an integer view so NaN payloads survive.
    mask = torch.rand(shape, device="cuda") < 0.3
    picks = torch.randint(0, len(pool), (int(mask.sum()),), device="cuda")
    x.view(view_dtype)[mask] = pool[picks]
    return x


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "SM90+ required")
@skipIfNoCuteDSL
class TestCuTeDSLRegisterSort(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        # The references are functional sorts, which the override also routes;
        # keep them on stock so the expectation is not the code under test.
        registry.deregister_op_overrides(disable_op_symbols="sort.stable")
        self.addCleanup(registry.reenable_op_overrides, enable_op_symbols="sort.stable")

    def _check(
        self,
        x: torch.Tensor,
        index_dtype: torch.dtype,
        descending: bool,
        values: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        from torch._native.ops.sort.cutedsl_register import register_sort

        if values is None:
            values = torch.empty_like(x)
        if indices is None:
            indices = torch.empty(x.shape, dtype=index_dtype, device=x.device)
        register_sort(x, values, indices, descending, _arch())
        ref_values, ref_indices = torch.sort(
            x, dim=-1, stable=True, descending=descending
        )
        self.assertEqual(values.view(torch.uint8), ref_values.view(torch.uint8))
        self.assertEqual(indices.long(), ref_indices)

    @parametrize("key_dtype", _NSORT_KEY_DTYPES)
    @parametrize("descending", (False, True))
    def test_nsorts(self, key_dtype, descending):
        # Each spec is a JIT compile: every length for ascending float32, the
        # leaf boundaries otherwise (test_nsorts_every_length has the rest).
        full = key_dtype == torch.float32 and not descending
        self._sweep(key_dtype, descending, _NSORTS if full else _BOUNDARY_NSORTS)

    @slowTest
    @parametrize("key_dtype", _NSORT_KEY_DTYPES)
    @parametrize("descending", (False, True))
    def test_nsorts_every_length(self, key_dtype, descending):
        self._sweep(key_dtype, descending, _NSORTS)

    def _sweep(self, key_dtype, descending, nsorts):
        torch.manual_seed(0)
        for nsort in nsorts:
            for rows in _ROWS:
                x = _make_keys(key_dtype, (rows, nsort))
                self._check(x, torch.int16, descending)

    # Every index dtype with float32, every key with int16 indices.
    @parametrize(
        "index_dtype,key_dtype",
        _slow_unless(
            lambda i, k: k == torch.float32 or i == torch.int16,
            _INDEX_DTYPES,
            _KEY_DTYPES,
        ),
    )
    def test_index_dtypes(self, index_dtype, key_dtype):
        torch.manual_seed(1)
        # The full index range for 8-bit indices; rung 2048 is in test_nsorts.
        limit = min(256, torch.iinfo(index_dtype).max + 1)
        for nsort, descending in ((100, True), (limit, False)):
            x = _make_keys(key_dtype, (37, nsort))
            self._check(x, index_dtype, descending)

    @parametrize(
        "key_dtype",
        _slow_unless(
            lambda k: k in (torch.uint8, torch.float32),
            (torch.uint8, torch.float16, torch.float32, torch.int64),
        ),
    )
    def test_misaligned(self, key_dtype):
        from torch._native.ops.sort.policy import register_store

        torch.manual_seed(2)
        # Element offsets of x, values and indices; each narrows the vector
        # width to what every pointer is aligned to.
        rows, nsort, pad = 33, 256, 4
        size = rows * nsort + pad
        base_x = _make_keys(key_dtype, (size,))
        base_values = torch.empty(size, dtype=key_dtype, device="cuda")
        base_indices = torch.empty(size, dtype=torch.int16, device="cuda")
        stores = set()
        for ox, ov, oi in itertools.product((0, 1, 2, 4), repeat=3):
            x = base_x[ox : ox + rows * nsort].view(rows, nsort)
            values = base_values[ov : ov + rows * nsort].view(rows, nsort)
            indices = base_indices[oi : oi + rows * nsort].view(rows, nsort)
            operands = [(t.data_ptr(), t.element_size()) for t in (x, values, indices)]
            stores.add(register_store(nsort, operands))
            for descending in (False, True):
                self._check(x, torch.int16, descending, values, indices)
        self.assertTrue({1, 2, 4} <= stores, stores)

    def test_empty(self):
        for shape in ((0, 64), (5, 0)):
            x = torch.empty(shape, device="cuda")
            self._check(x, torch.int16, False)


instantiate_parametrized_tests(TestCuTeDSLRegisterSort)


if __name__ == "__main__":
    run_tests()
