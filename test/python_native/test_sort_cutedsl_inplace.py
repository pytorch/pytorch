# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the CuTeDSL in-place sort leaves (rows of at most 4096).
# Each kernel mirrors the stock leaf for its row length and must match it
# bit-exactly on values and indices, including the leaf's order of special
# values: the comparison leaves (n <= 128) tie all NaNs above +inf, the radix
# leaf puts negative NaN first; both tie -0.0 with +0.0. Key widths are 1, 2, 4
# and 8 bytes.

import unittest

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


_KEY_DTYPES = (
    torch.bool,
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
# Each side of every leaf boundary: 32 (bitonic), 128 (warp merge), 4096.
_NSORTS = (1, 2, 31, 32, 33, 127, 128, 129, 4095, 4096)
_NEG_NAN = {
    torch.float16: 0xFE00,
    torch.bfloat16: 0xFFC0,
    torch.float32: 0xFFC00000,
    torch.float64: 0xFFF8000000000000,
}
_ROWS = 37


def _arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


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

    @parametrize("key_dtype", _KEY_DTYPES)
    @parametrize("descending", (False, True))
    @parametrize("stable", (False, True))
    def test_leaf_boundaries(self, key_dtype, descending, stable):
        for nsort in _NSORTS:
            x = _make_keys(key_dtype, (_ROWS, nsort))
            self._check(x, torch.int16, descending, stable)

    @parametrize("index_dtype", _INDEX_DTYPES)
    @parametrize("key_dtype", (torch.bool, torch.bfloat16, torch.int32, torch.float64))
    def test_index_dtypes(self, index_dtype, key_dtype):
        limit = min(4096, torch.iinfo(index_dtype).max + 1)
        for nsort, stable in ((32, False), (min(128, limit), True), (limit, True)):
            for descending in (False, True):
                x = _make_keys(key_dtype, (_ROWS, nsort))
                self._check(x, index_dtype, descending, stable)

    def test_many_rows(self):
        for nsort, stable in ((17, False), (100, True), (300, True), (2000, True)):
            x = _make_keys(torch.float16, (4099, nsort))
            self._check(x, torch.uint16, False, stable)

    def test_empty(self):
        for shape in ((0, 64), (5, 0)):
            x = torch.empty(shape, device="cuda")
            self._check(x, torch.int16, False, True)


instantiate_parametrized_tests(TestCuTeDSLInplaceSort)


if __name__ == "__main__":
    run_tests()
