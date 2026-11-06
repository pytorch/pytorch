# Owner(s): ["module: dsl-native-ops"]
#
# Correctness tests for the CuTeDSL device-wide sort leaves (per-row onesweep,
# full sort, segmented radix sort) that write narrow out= indices. Each must
# match the stock CUDA sort bit-exactly on values and indices.

import itertools
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
    slowTest,
    subtest,
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
_NEG_NAN = {
    torch.float16: 0xFE00,
    torch.bfloat16: 0xFFC0,
    torch.float32: 0xFFC00000,
    torch.float64: 0xFFF8000000000000,
}
_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


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
    if dtype.is_floating_point:
        # Few distinct values give many ties; specials exercise the ordinal map.
        x = torch.randint(-8, 8, shape, device="cuda").to(dtype) / 4
        specials = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan")], dtype=dtype
        )
        x[:, :5] = specials.to("cuda")
        bits = _NEG_NAN[dtype]
        bits = (
            bits - (1 << (8 * dtype.itemsize))
            if bits >> (8 * dtype.itemsize - 1)
            else bits
        )
        neg_nan = torch.tensor([bits], dtype=_INT_VIEW[dtype.itemsize])
        x[:, 5] = neg_nan.view(dtype).to("cuda")
        return x
    info = torch.iinfo(dtype)
    x = torch.randint(
        max(info.min, -100), min(info.max, 100) + 1, shape, device="cuda", dtype=dtype
    )
    extremes = torch.tensor(
        [info.min, info.min + 1, info.max - 1, info.max, 0], dtype=dtype
    )
    x[:, :5] = extremes.to("cuda")
    return x


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "SM90+ required")
@skipIfNoCuteDSL
class TestCuTeDSLGlobalSort(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        # The references are functional sorts, which the override also routes;
        # keep them on stock so the expectation is not the code under test.
        registry.deregister_op_overrides(disable_op_symbols="sort.stable")
        self.addCleanup(registry.reenable_op_overrides, enable_op_symbols="sort.stable")

    def _check(
        self, x: torch.Tensor, index_dtype: torch.dtype, descending: bool
    ) -> None:
        from torch._native.ops.sort.cutedsl_global import global_sort

        values = torch.empty_like(x)
        indices = torch.empty(x.shape, dtype=index_dtype, device=x.device)
        global_sort(x, values, indices, descending, _arch())
        ref_values, ref_indices = torch.sort(
            x, dim=-1, stable=True, descending=descending
        )
        self.assertEqual(values.view(torch.uint8), ref_values.view(torch.uint8))
        self.assertEqual(indices.long(), ref_indices)

    # One shape per leaf: one row, fewer than 128 rows, segmented rows.
    @parametrize("key_dtype", _slow_unless(lambda k: k in _FAST_KEYS, _KEY_DTYPES))
    @parametrize("descending", (False, True))
    @parametrize("rows", (1, 3, 130))
    def test_key_dtypes(self, key_dtype, descending, rows):
        self._check(_make_keys(key_dtype, (rows, 6000)), torch.int32, descending)

    # Every index dtype with float32, every key with uint16; the segmented leaf
    # (128 rows) compiles slowest, so one pair there.
    @parametrize(
        "key_dtype,index_dtype,rows",
        _slow_unless(
            lambda k, i, r: (k == torch.float32 or i == torch.uint16)
            and (r < 128 or (k, i) == (torch.float32, torch.int16)),
            (torch.bool, torch.float16, torch.float32, torch.int64),
            (torch.uint16, torch.int16, torch.uint32),
            (1, 3, 128),
        ),
    )
    def test_index_dtypes(self, key_dtype, index_dtype, rows):
        self._check(_make_keys(key_dtype, (rows, 4097)), index_dtype, False)

    @parametrize("rows", (1, 3, 127, 128, 300))
    @parametrize("nsort", (4097, 6000, 70000, 200000))
    def test_shapes(self, rows, nsort):
        for key_dtype in (torch.float16, torch.int32):
            x = _make_keys(key_dtype, (rows, nsort))
            # Small non-negative keys in row 0: one digit in a pass takes the
            # segmented leaf's copy branch (cub's AgentRadixSortDownsweep::Copy).
            x[0] = torch.randint(0, 100, (nsort,), device="cuda").to(key_dtype)
            self._check(x, torch.int32, False)

    @parametrize("rows", (1, 2))
    def test_large_segments(self, rows):
        for key_dtype in (torch.bool, torch.float32):
            x = _make_keys(key_dtype, (rows, 1_000_001))
            self._check(x, torch.int32, True)

    @largeTensorTest("12GB", device="cuda")
    def test_long_row_portions(self):
        # Past 2^31 - 2^28 elements the portion count once wrapped Int32.
        n = 2**31 - 2**28 + 2**20
        x = torch.randint(0, 256, (1, n), dtype=torch.uint8, device="cuda")
        values = torch.empty_like(x)
        indices = torch.empty(1, n, dtype=torch.int32, device="cuda")
        torch.sort(x, stable=True, out=(values, indices))
        self.assertTrue(bool((values[0, 1:] >= values[0, :-1]).all()))
        sample = indices[0, :: 2**20].long()
        self.assertEqual(x[0, sample], values[0, :: 2**20])
        del x, values, indices

    def test_histogram_grid_larger_than_row(self):
        # The histogram loop's counter once wrapped Int32 when a block's last
        # tile plus the grid step passed 2^31 - 1; a huge grid on a short row
        # exercises the same step without 2^31 elements.
        from torch._native.cutedsl.launch import compile_kernel
        from torch._native.ops.sort import cutedsl_global as g

        x = _make_keys(torch.uint8, (1, 100_000))
        spec = g.onesweep_spec(_arch(), "unsigned", 8, "int32", False)
        sorter, fake = g.compile_args(spec)
        run = compile_kernel(sorter, *fake)
        values = torch.empty_like(x)
        indices = torch.empty(x.shape, dtype=torch.int32, device="cuda")
        state = torch.empty(
            sorter.state_words(x.numel()), dtype=torch.int32, device="cuda"
        )
        # One pass for 8-bit keys: the outputs double as the pass buffers.
        run(
            x[0],
            values[0],
            values[0],
            indices[0],
            indices[0],
            state,
            1 << 20,
            x.shape[1],
            0,
        )
        ref_values, ref_indices = torch.sort(x, stable=True)
        self.assertEqual(values, ref_values)
        self.assertEqual(indices.long(), ref_indices)

    def test_portions(self):
        # cub carries each portion's bins into the next; shrink the portion
        # to a few tiles to cover that path.
        from torch._native.cutedsl.launch import compile_kernel
        from torch._native.ops.sort import cutedsl_global as g

        x = _make_keys(torch.float32, (1, 100_000))
        spec = g.onesweep_spec(_arch(), "float", 32, "int32", False)
        tile = g._sorter(spec).tile
        with mock.patch.object(g, "_PORTION_LIMIT", 3 * tile + 1):
            sorter, fake = g.compile_args(spec)
            self.assertGreater(-(-x.numel() // sorter.portion), 1)
            run = compile_kernel(sorter, *fake)
        values = torch.empty_like(x)
        indices = torch.empty(x.shape, dtype=torch.int32, device="cuda")
        state = torch.empty(
            sorter.state_words(x.numel()), dtype=torch.int32, device="cuda"
        )
        keys_tmp = torch.empty(x.numel(), dtype=torch.uint32, device="cuda")
        idx_tmp = torch.empty(x.numel(), dtype=torch.int32, device="cuda")
        blocks = g._hist_blocks(x.device, _arch())
        run(
            x.view(torch.uint32)[0],
            keys_tmp,
            values.view(torch.uint32)[0],
            idx_tmp,
            indices[0],
            state,
            blocks,
            x.shape[1],
            0,
        )
        ref_values, ref_indices = torch.sort(x, dim=-1, stable=True)
        self.assertEqual(values.view(torch.uint8), ref_values.view(torch.uint8))
        self.assertEqual(indices.long(), ref_indices)


instantiate_parametrized_tests(TestCuTeDSLGlobalSort)


if __name__ == "__main__":
    run_tests()
