# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the CuTeDSL in-place sort leaves at the rows the router
# sends them: unstable rows of at most 32 (bitonic) and rows of 2049 .. 4096
# (radix). Each kernel mirrors the stock leaf and must match it bit-exactly on
# values and indices, including the leaf's order of special values: the bitonic
# leaf ties all NaNs above +inf, the radix leaf puts negative NaN first; both tie
# -0.0 with +0.0. Key widths are 1, 2, 4 and 8 bytes.

import itertools
import unittest

import torch
from torch._native import registry
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import largeTensorTest
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


# bool keys take the partition kernel.
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
# (nsort, stable) the router sends here: unstable rows of at most 32 (bitonic)
# and rows of 2049 .. 4096 (radix), each side of both boundaries.
_LEAF_ROWS = (
    *((n, False) for n in (1, 2, 31, 32)),
    *((n, s) for n in (2049, 4095, 4096) for s in (False, True)),
)
_NEG_NAN = {
    torch.float16: 0xFE00,
    torch.bfloat16: 0xFFC0,
    torch.float32: 0xFFC00000,
    torch.float64: 0xFFF8000000000000,
}
_ROWS = 37


# Each JIT compile costs seconds, so the rest of each cross is @slowTest. The
# fast set is all CI runs: every key kind (unsigned, signed, float) on each side
# of the 32-bit sign wrap of the padding key, in both directions.
_FAST_KEYS = (torch.uint8, torch.int8, torch.bfloat16, torch.float32, torch.int64)


def _slow_unless(fast, *axes):
    """Every combination of ``axes``; those ``fast`` rejects are @slowTest."""
    cases = [c[0] if len(axes) == 1 else c for c in itertools.product(*axes)]
    fast_ = fast if len(axes) == 1 else lambda c: fast(*c)
    return [c if fast_(c) else subtest(c, decorators=[slowTest]) for c in cases]


def _arch() -> int:
    from torch._native.ops.sort.cutedsl_impl import _arch

    return _arch(torch.cuda.current_device())


def _make_keys(dtype: torch.dtype, shape: tuple[int, int]) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device="cuda").bool()
    # Few distinct values give many ties; a third of the keys are specials.
    if dtype.is_floating_point:
        x = torch.randint(-8, 8, shape, device="cuda").to(dtype) / 4
        int_dtype = {2: torch.int16, 4: torch.int32, 8: torch.int64}[dtype.itemsize]
        neg_nan = _NEG_NAN[dtype] - (1 << (8 * dtype.itemsize))
        specials = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")])
        neg_nan = torch.tensor([neg_nan], dtype=int_dtype).view(dtype)
        pool = torch.cat([specials.to(dtype), neg_nan]).to("cuda")
    else:
        info = torch.iinfo(dtype)
        lo, hi = max(info.min, -100), min(info.max, 100)
        x = torch.randint(lo, hi + 1, shape, device="cuda", dtype=dtype)
        extremes = (info.min, info.min + 1, info.max - 1, info.max)
        pool = torch.tensor(extremes, dtype=dtype, device="cuda")
    mask = torch.rand(shape, device="cuda") < 0.3
    x[mask] = pool[torch.randint(0, len(pool), (int(mask.sum()),), device="cuda")]
    return x


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "SM90+ required")
@skipIfNoCuteDSL
class TestCuTeDSLInplaceSort(TestCase):
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
        stable: bool,
    ) -> None:
        from torch._native.ops.sort.cutedsl_inplace import inplace_sort

        values = torch.empty_like(x)
        indices = torch.empty(x.shape, dtype=index_dtype, device=x.device)
        inplace_sort(x, values, indices, descending, stable, _arch())
        ref_values, ref_indices = torch.sort(
            x, dim=-1, stable=stable, descending=descending
        )
        self.assertEqual(values.view(torch.uint8), ref_values.view(torch.uint8))
        # The bitonic leaf is unstable but deterministic: its network is mirrored.
        self.assertEqual(indices.long(), ref_indices)

    @parametrize("key_dtype", _slow_unless(lambda k: k in _FAST_KEYS, _KEY_DTYPES))
    @parametrize("descending", (False, True))
    def test_leaf_boundaries(self, key_dtype, descending):
        for nsort, stable in _LEAF_ROWS:
            x = _make_keys(key_dtype, (_ROWS, nsort))
            self._check(x, torch.int16, descending, stable)

    # Every index dtype with int32 keys, every key with int16 indices.
    @parametrize(
        "index_dtype,key_dtype",
        _slow_unless(
            lambda i, k: k == torch.int32 or i == torch.int16,
            _INDEX_DTYPES,
            (torch.uint8, torch.bfloat16, torch.int32, torch.float64),
        ),
    )
    def test_index_dtypes(self, index_dtype, key_dtype):
        # 8-bit indices cannot address the radix leaf's rows.
        limit = min(4096, torch.iinfo(index_dtype).max + 1)
        rows = [(32, False)] + ([(limit, True)] if limit > 2048 else [])
        for nsort, stable in rows:
            for descending in (False, True):
                x = _make_keys(key_dtype, (_ROWS, nsort))
                self._check(x, index_dtype, descending, stable)

    def test_many_rows(self):
        for nsort, stable in ((17, False), (3000, True)):
            x = _make_keys(torch.float16, (4099, nsort))
            self._check(x, torch.uint16, False, stable)

    @largeTensorTest("7GB", device="cuda")
    def test_max_rows(self):
        # policy.MAX_ROWS rows: the grid count must not wrap Int32.
        from torch._native.ops.sort.policy import MAX_ROWS

        x = torch.zeros(MAX_ROWS, 1, dtype=torch.uint8, device="cuda")
        values, indices = torch.empty_like(x), torch.empty_like(x)
        for stable in (False, True):
            torch.sort(x, stable=stable, out=(values, indices))
            self.assertEqual(int(indices.max()), 0)

    def test_empty(self):
        for shape in ((0, 64), (5, 0)):
            x = torch.empty(shape, device="cuda")
            self._check(x, torch.int16, False, True)


instantiate_parametrized_tests(TestCuTeDSLInplaceSort)


if __name__ == "__main__":
    run_tests()
