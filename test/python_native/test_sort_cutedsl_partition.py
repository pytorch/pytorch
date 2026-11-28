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
from torch._native import registry
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import largeTensorTest
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
# Row lengths; 256 and 32768 are the longest that uint8 and int16 indices address.
# 17, 4097 and 65537 end the input inside a 16-byte chunk.
_NS = (
    1,
    2,
    17,
    31,
    32,
    33,
    127,
    128,
    129,
    256,
    4096,
    4097,
    10677,
    32768,
    65536,
    65537,
    200000,
)
_ROWS = (1, 3, 127, 128, 1000)


def _arch() -> int:
    from torch._native.ops.sort.cutedsl_impl import _arch

    return _arch(torch.cuda.current_device())


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "SM90+ required")
@skipIfNoCuteDSL
class TestCuTeDSLPartitionSort(TestCase):
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
    ) -> set[str]:
        """Sorts ``x`` and returns the launch modes taken."""
        from torch._native.ops.sort import cutedsl_partition

        if values is None:
            values = torch.empty_like(x)
        if indices is None:
            indices = torch.empty(x.shape, dtype=index_dtype, device=x.device)
        with mock.patch.object(
            cutedsl_partition, "make_spec", wraps=cutedsl_partition.make_spec
        ) as make_spec:
            cutedsl_partition.partition_sort(x, values, indices, descending, _arch())
        ref_values, ref_indices = torch.sort(
            x, dim=-1, stable=True, descending=descending
        )
        self.assertEqual(values, ref_values)
        self.assertEqual(indices.long(), ref_indices)
        return {call.args[3] for call in make_spec.call_args_list}

    def _row_mode_rows(self) -> int:
        """Fewest rows that take a CTA per row on this GPU."""
        from torch._native.cutedsl.hw_caps import caps
        from torch._native.ops.sort.cutedsl_partition import policy

        threads = policy(torch.device("cuda")).row_threads
        return caps(torch.device("cuda")).fill_blocks(threads, waves=2.0)

    @parametrize("index_dtype", _INDEX_DTYPES)
    @parametrize("descending", (False, True))
    def test_shapes(self, index_dtype, descending):
        torch.manual_seed(0)
        limit = torch.iinfo(index_dtype).max + 1
        for nsort in _NS:
            if nsort > limit:
                continue
            for rows in _ROWS:
                if rows * nsort > 100_000_000:
                    continue  # test_large
                x = torch.rand(rows, nsort, device="cuda") < 0.5
                # All-false and all-true rows.
                x[0] = False
                x[-1] = True
                self._check(x, index_dtype, descending)
                del x

    @parametrize("index_dtype", (torch.uint16, torch.int32))
    @parametrize("descending", (False, True))
    def test_modes(self, index_dtype, descending):
        torch.manual_seed(0)
        fill = self._row_mode_rows()
        # Rows past one ``rows`` tile (1024 16-byte chunks) take ``row`` with
        # enough rows to fill the GPU, else ``split``.
        cases = [
            ("rows", 1000, 129),
            ("rows", 3, 10677),
            ("row", fill, 20000),
            ("row", fill + 3, 30000),
            ("split", fill - 1, 20000),
            ("split", 3, 65535),
        ]
        for mode, rows, nsort in cases:
            x = torch.rand(rows, nsort, device="cuda") < 0.5
            x[0] = False
            x[-1] = True
            self.assertEqual(self._check(x, index_dtype, descending), {mode})

    @largeTensorTest("6GB", device="cuda")
    def test_large(self):
        torch.manual_seed(0)
        x = torch.rand(1000, 200000, device="cuda") < 0.5
        for index_dtype in (torch.uint32, torch.int32):
            self._check(x, index_dtype, True)

    @largeTensorTest("8GB", device="cuda")
    def test_rows_int32_limit(self):
        # Row counts near the Int32 limit: one launch, then two row batches.
        from torch._native.ops.sort import cutedsl_partition
        from torch._native.ops.sort.policy import PARTITION_MAX_NSORT

        for rows in (PARTITION_MAX_NSORT, (1 << 31) - 1):
            x = torch.zeros(rows, 1, dtype=torch.bool, device="cuda")
            x[::3] = True
            values = torch.empty_like(x)
            indices = torch.full(x.shape, 1, dtype=torch.uint8, device="cuda")
            cutedsl_partition.partition_sort(x, values, indices, False, _arch())
            self.assertTrue(torch.equal(values, x))
            self.assertFalse(indices.any())
            del x, values, indices

    @largeTensorTest("13GB", device="cuda")
    def test_longest_row(self):
        # Tile extents past the row end stay in Int32; offset 1 forces 1-byte
        # chunks. Checked in slices: a stock reference would not fit.
        from torch._native.ops.sort import cutedsl_partition
        from torch._native.ops.sort.policy import PARTITION_MAX_NSORT

        n, step = PARTITION_MAX_NSORT, 1 << 26
        for offset in (0, 1):
            x = (torch.rand(n + offset, device="cuda", dtype=torch.half) < 0.5)[offset:]
            values = torch.empty_like(x)
            indices = torch.full((n,), -1, dtype=torch.int32, device="cuda")
            cutedsl_partition.partition_sort(
                x[None], values[None], indices[None], False, _arch()
            )
            falses = n - sum(int(x[s : s + step].sum()) for s in range(0, n, step))
            for s in range(0, n, step):
                idx = indices[s : s + step + 1].long()
                pos = torch.arange(s, s + idx.numel(), device="cuda")
                self.assertTrue(torch.equal(values[s : s + step], pos[:step] >= falses))
                self.assertTrue(torch.equal(x[idx], pos >= falses))
                rising = (idx[1:] > idx[:-1]) | (pos[1:] == falses)
                self.assertTrue(bool(rising.all()) and bool((idx >= 0).all()))
            del x, values, indices

    @parametrize("descending", (False, True))
    def test_skewed(self, descending):
        torch.manual_seed(0)
        for p in (0.001, 0.999):
            for rows, nsort in ((4099, 100), (37, 4097), (600, 65536), (2, 200000)):
                x = torch.rand(rows, nsort, device="cuda") < p
                self._check(x, torch.int32, descending)

    @parametrize("index_dtype", (torch.uint8, torch.uint16, torch.int32))
    @parametrize("descending", (False, True))
    def test_unaligned(self, index_dtype, descending):
        # Element offsets of keys, values and indices that defeat their 16-byte
        # chunks one at a time and together; uint8 and uint16 indices at offset
        # 0 store 16 and 8 per vector.
        torch.manual_seed(0)
        limit = torch.iinfo(index_dtype).max + 1
        offsets = ((1, 0, 0), (0, 3, 0), (0, 0, 1), (0, 0, 3), (8, 1, 2), (3, 3, 3))
        for rows, nsort in ((33, 31), (7, 256), (5, 4097), (3, 65536), (3, 70000)):
            if nsort > limit:
                continue
            size = rows * nsort
            for ox, ov, oi in offsets:
                x = torch.rand(size + ox, device="cuda") < 0.5
                values = torch.empty(size + ov, dtype=torch.bool, device="cuda")
                indices = torch.empty(size + oi, dtype=index_dtype, device="cuda")
                x = x[ox:].view(rows, nsort)
                values = values[ov:].view(rows, nsort)
                indices = indices[oi:].view(rows, nsort)
                self._check(x, index_dtype, descending, values, indices)

    def test_row_batches(self):
        # Inputs past the Int32 flat extent are sorted in row batches.
        torch.manual_seed(0)
        from torch._native.ops.sort import cutedsl_partition

        with mock.patch.object(cutedsl_partition, "PARTITION_MAX_NSORT", 1000):
            for nsort in (100, 999, 4097):
                x = torch.rand(37, nsort, device="cuda") < 0.5
                self._check(x, torch.int16, True)

    def test_empty(self):
        for shape in ((0, 64), (5, 0)):
            x = torch.empty(shape, dtype=torch.bool, device="cuda")
            self.assertEqual(self._check(x, torch.int16, False), set())


instantiate_parametrized_tests(TestCuTeDSLPartitionSort)


if __name__ == "__main__":
    run_tests()
