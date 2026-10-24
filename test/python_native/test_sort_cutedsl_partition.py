# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the CuTeDSL partition leaf: the stable sort of bool
# keys as a counting partition. The stable order is unique, so values and
# indices must match stock ``torch.sort(stable=True)`` bit-exactly. Shapes
# cover every launch mode: several rows per CTA, a CTA per row and rows split
# over CTAs, with rows that straddle the 16-byte chunks.

import unittest
from unittest import mock

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


_INDEX_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.uint16,
    torch.int16,
    torch.uint32,
    torch.int32,
)
_NSORTS = (1, 2, 31, 32, 33, 127, 128, 129, 4096, 4097, 10677, 65536, 200000)
_ROWS = (1, 3, 127, 128, 1000)


def _arch() -> int:
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "SM90+ required")
@skipIfNoCuteDSL
class TestCuTeDSLPartitionSort(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _check(
        self,
        x: torch.Tensor,
        index_dtype: torch.dtype,
        descending: bool,
        values: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> None:
        from torch._native.ops.sort.cutedsl_partition import partition_sort

        if values is None:
            values = torch.empty_like(x)
        if indices is None:
            indices = torch.empty(x.shape, dtype=index_dtype, device=x.device)
        partition_sort(x, values, indices, descending, _arch())
        ref_values, ref_indices = torch.sort(
            x, dim=-1, stable=True, descending=descending
        )
        self.assertEqual(values, ref_values)
        self.assertEqual(indices.long(), ref_indices)

    @parametrize("index_dtype", _INDEX_DTYPES)
    @parametrize("descending", (False, True))
    def test_shapes(self, index_dtype, descending):
        limit = torch.iinfo(index_dtype).max + 1
        for nsort in _NSORTS:
            if nsort > limit:
                continue
            for rows in _ROWS:
                x = torch.rand(rows, nsort, device="cuda") < 0.5
                # All-false and all-true rows.
                x[0] = False
                x[-1] = True
                self._check(x, index_dtype, descending)
                del x

    @parametrize("descending", (False, True))
    def test_skewed(self, descending):
        for p in (0.001, 0.999):
            for rows, nsort in ((4099, 100), (37, 4097), (600, 65536), (2, 200000)):
                x = torch.rand(rows, nsort, device="cuda") < p
                self._check(x, torch.int32, descending)

    def test_unaligned(self):
        # Offsets that defeat the 16-byte chunks of keys, values and indices.
        for rows, nsort in ((33, 31), (5, 4097), (3, 70000)):
            size = rows * nsort
            for offset in (1, 3, 8):
                x = (torch.rand(size + offset, device="cuda") < 0.5)[offset:]
                x = x.view(rows, nsort)
                values = torch.empty(size + offset, dtype=torch.bool, device="cuda")
                indices = torch.empty(size + offset, dtype=torch.int32, device="cuda")
                values = values[offset:].view(rows, nsort)
                indices = indices[offset:].view(rows, nsort)
                self._check(x, torch.int32, True, values, indices)

    def test_row_batches(self):
        # Inputs past the Int32 flat extent are sorted in row batches.
        with mock.patch("torch._native.ops.sort.cutedsl_partition._FLAT_LIMIT", 1000):
            for nsort in (100, 999, 4097):
                x = torch.rand(37, nsort, device="cuda") < 0.5
                self._check(x, torch.int16, True)

    def test_empty(self):
        for shape in ((0, 64), (5, 0)):
            x = torch.empty(shape, dtype=torch.bool, device="cuda")
            self._check(x, torch.int16, False)


instantiate_parametrized_tests(TestCuTeDSLPartitionSort)


if __name__ == "__main__":
    run_tests()
