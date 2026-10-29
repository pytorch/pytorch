# Owner(s): ["module: dsl-native-ops"]
#
# Routing tests for the CuTeDSL overrides of aten::sort.values_stable and
# aten::sort.stable: they fire for every out= call with narrow indices, for
# functional sort and for int64 out=, on every stock leaf, pick the kernel
# _launcher documents, and the result always matches the stock sort.

import collections
import contextlib
import unittest
from unittest import mock

import torch
from torch._native import registry
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "SM90+ required")
@skipIfNoCuteDSL
class TestCuTeDSLSortOverride(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        # These test the JIT route; in a build with embedded kernels, covered
        # calls would never reach it (test_sort_native_aot covers those).
        aot_enabled = torch._native.aot_enabled()
        torch._native.set_aot_enabled(False)
        self.addCleanup(torch._native.set_aot_enabled, aot_enabled)
        # The references are functional sorts, which the override also routes:
        # keep them on stock, and let _functional_routed turn it back on.
        registry.deregister_op_overrides(disable_op_symbols="sort.stable")
        self.addCleanup(registry.reenable_op_overrides, enable_op_symbols="sort.stable")

    @contextlib.contextmanager
    def _functional_routed(self):
        registry.reenable_op_overrides(enable_op_symbols="sort.stable")
        try:
            yield
        finally:
            registry.deregister_op_overrides(disable_op_symbols="sort.stable")

    def _kernels(self, fn) -> collections.Counter:
        """Calls of each top-level kernel entry point while running fn."""
        from torch._native.ops.sort import (
            cutedsl_global,
            cutedsl_inplace,
            cutedsl_partition,
            cutedsl_register,
        )

        entries = (
            (cutedsl_inplace, "inplace_sort"),
            (cutedsl_global, "global_sort"),
            (cutedsl_register, "register_sort"),
            (cutedsl_partition, "partition_sort"),
        )
        with contextlib.ExitStack() as stack:
            mocks = {
                name: stack.enter_context(
                    mock.patch.object(module, name, wraps=getattr(module, name))
                )
                for module, name in entries
            }
            fn()
        return collections.Counter(
            {name: m.call_count for name, m in mocks.items() if m.call_count}
        )

    def _launches(self, fn) -> int:
        return sum(self._kernels(fn).values())

    @parametrize("dim", (0, 1, -1))
    @parametrize("descending", (False, True))
    def test_out_narrow_fires_and_matches(self, dim, descending):
        x = (torch.randn(200, 200, 30, device="cuda") * 4).half()
        x = x.reshape(6000, 200).T.contiguous() if dim != 0 else x.reshape(6000, 200)
        ref_values, ref_indices = torch.sort(
            x, dim=dim, stable=True, descending=descending
        )
        values = torch.empty(0, dtype=x.dtype, device="cuda")
        indices = torch.empty(0, dtype=torch.int16, device="cuda")
        out = (values, indices)
        launches = self._launches(
            lambda: torch.sort(x, dim=dim, stable=True, descending=descending, out=out)
        )
        self.assertEqual(launches, 1)
        self.assertEqual(values, ref_values)
        self.assertEqual(indices.long(), ref_indices)

    # Expected entry point per (row length, stable); written out rather than
    # derived, so a wrong rule in _launcher cannot also be the expectation.
    _SHORT_ROWS = {
        (16, False): "inplace_sort",  # bitonic leaf beats register here
        (16, True): "register_sort",
        (32, False): "inplace_sort",
        (32, True): "register_sort",
        (33, False): "register_sort",
        (33, True): "register_sort",
        (100, False): "register_sort",
        (2048, False): "register_sort",
        (2048, True): "register_sort",
        (2049, False): "inplace_sort",
        (2049, True): "inplace_sort",
        (3000, True): "inplace_sort",
    }

    @parametrize(
        "case",
        list(_SHORT_ROWS),
        name_fn=lambda c: f"n{c[0]}_{'stable' if c[1] else 'unstable'}",
    )
    @parametrize("descending", (False, True))
    def test_short_rows_pick_kernel(self, case, descending):
        nsort, stable = case
        torch.manual_seed(nsort)
        x = torch.randn(1000, nsort, device="cuda")
        indices = torch.empty(0, dtype=torch.int16, device="cuda")
        out = (torch.empty_like(x), indices)
        kernels = self._kernels(
            lambda: torch.sort(x, stable=stable, descending=descending, out=out)
        )
        self.assertEqual(kernels, {self._SHORT_ROWS[case]: 1})
        ref_values, ref_indices = torch.sort(x, stable=True, descending=descending)
        self.assertEqual(out[0], ref_values)
        self.assertEqual(indices.long(), ref_indices)

    @parametrize("dim", (0, -1))
    def test_register_paths(self, dim):
        # argsort and a non-last dim both reach the register kernel.
        torch.manual_seed(0)
        x = torch.randn(1000, 100, device="cuda")
        x = x.T.contiguous() if dim == 0 else x
        indices = torch.empty(0, dtype=torch.uint8, device="cuda")
        kernels = self._kernels(
            lambda: torch.argsort(x, dim=dim, stable=True, out=indices)
        )
        self.assertEqual(kernels, {"register_sort": 1})
        self.assertEqual(indices.long(), torch.argsort(x, dim=dim, stable=True))

    @parametrize(
        "shape, dim, index_dtype",
        [
            ((5000, 16), -1, torch.uint8),
            ((1000, 100), -1, torch.uint8),
            ((8000, 5), 0, torch.uint16),
            ((64, 70000), -1, torch.int32),
        ],
        name_fn=lambda shape,
        dim,
        index_dtype: f"{shape[0]}x{shape[1]}_dim{dim}_{str(index_dtype)[6:]}",
    )
    @parametrize("stable", (False, True))
    @parametrize("descending", (False, True))
    def test_bool_routes_to_partition(
        self, shape, dim, index_dtype, stable, descending
    ):
        torch.manual_seed(shape[1])
        x = torch.randint(0, 2, shape, device="cuda").bool()
        values = torch.empty(0, dtype=torch.bool, device="cuda")
        indices = torch.empty(0, dtype=index_dtype, device="cuda")
        call = dict(
            dim=dim, stable=stable, descending=descending, out=(values, indices)
        )
        kernels = self._kernels(lambda: torch.sort(x, **call))
        self.assertEqual(kernels, {"partition_sort": 1})
        ref_values, ref_indices = torch.sort(
            x, dim=dim, stable=True, descending=descending
        )
        self.assertEqual(values, ref_values)
        self.assertEqual(indices.long(), ref_indices)

    @parametrize(
        "shape, key",
        [((64, 1024), torch.bool), ((1, 8192), torch.float16)],
        name_fn=lambda shape, key: f"{shape[0]}x{shape[1]}_{str(key)[6:]}",
    )
    def test_short_calls_stay_on_stock(self, shape, key):
        # policy.jit_pays_off: one stock kernel beats the JIT route's host cost.
        for rows, launches in ((shape[0], 0), (shape[0] * 2, 1)):
            x = torch.randn(rows, shape[1], device="cuda").to(key)
            out = (
                torch.empty_like(x),
                torch.empty(x.shape, dtype=torch.uint16, device="cuda"),
            )
            self.assertEqual(
                self._launches(lambda: torch.sort(x, stable=True, out=out)), launches
            )

    def test_long_bool_rows_go_global(self):
        # Partition extents are Int32; longer bool rows take the mirror kernels.
        from torch._native.ops.sort import cutedsl_global, cutedsl_impl
        from torch._native.ops.sort.policy import PARTITION_MAX_NSORT

        for nsort, expected in (
            (PARTITION_MAX_NSORT, "partition_sort"),
            (PARTITION_MAX_NSORT + 1, "global_sort"),
        ):
            x = torch.zeros(1, 1, dtype=torch.bool, device="cuda").expand(1, nsort)
            launch = cutedsl_impl._launcher(x, nsort, stable=True, descending=False)
            self.assertEqual(launch.func.__name__, expected)
        self.assertIs(launch.func, cutedsl_global.global_sort)

    def test_argsort_out_fires(self):
        x = torch.randint(0, 2, (256, 8000), device="cuda").bool()
        indices = torch.empty(0, dtype=torch.uint16, device="cuda")
        kernels = self._kernels(lambda: torch.argsort(x, stable=True, out=indices))
        self.assertEqual(kernels, {"partition_sort": 1})
        self.assertEqual(indices.long(), torch.argsort(x, stable=True))

    # One shape per leaf the int64 route reaches:
    # (shape, key dtype, stable, entry point).
    _INT64_LEAVES = {
        "register": ((4096, 128), torch.float32, True, "register_sort"),
        "bitonic": ((16384, 32), torch.float32, False, "inplace_sort"),
        "radix": ((256, 3000), torch.bfloat16, True, "inplace_sort"),
        "partition": ((512, 1024), torch.bool, True, "partition_sort"),
        "full_sort": ((64, 8000), torch.float32, True, "global_sort"),
        "per_row": ((1, 600000), torch.float32, True, "global_sort"),
        "segmented": ((130, 5000), torch.int64, True, "global_sort"),
    }

    @parametrize("leaf", list(_INT64_LEAVES))
    @parametrize("functional", (True, False))
    def test_int64_fires_and_matches(self, leaf, functional):
        shape, key, stable, entry = self._INT64_LEAVES[leaf]
        torch.manual_seed(0)
        x = torch.randn(*shape, device="cuda")
        x = x > 0 if key == torch.bool else (x * 50).to(key)
        out = (
            torch.empty_like(x),
            torch.empty(shape, dtype=torch.int64, device="cuda"),
        )
        result = []

        def run():
            if functional:
                result.extend(torch.sort(x, stable=stable))
            else:
                result.extend(torch.sort(x, stable=stable, out=out))

        with self._functional_routed():
            kernels = self._kernels(run)
        self.assertEqual(kernels, {entry: 1})
        values, indices = result
        ref_values, ref_indices = torch.sort(x, stable=stable)
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(values, ref_values)
        self.assertEqual(indices, ref_indices)

    def test_short_int64_calls_fire(self):
        # No size-based stock fallback for int64 yet (see the TODO in _eligible),
        # including the calls jit_pays_off sends to stock for narrow indices.
        for x in (
            torch.randn(4, 128, device="cuda"),
            torch.randint(0, 2, (64, 1024), device="cuda").bool(),
        ):
            out = (torch.empty_like(x), torch.empty_like(x, dtype=torch.int64))
            with self._functional_routed():
                self.assertEqual(self._launches(lambda: torch.sort(x, stable=True)), 1)
            self.assertEqual(
                self._launches(lambda: torch.sort(x, stable=True, out=out)), 1
            )
            self.assertEqual(out[1], torch.sort(x, stable=True).indices)

    @parametrize("nsort, narrow", ((5000, torch.uint16), (70000, torch.int32)))
    def test_segmented_int64_sorts_narrow_index(self, nsort, narrow):
        # The segmented leaf sorts the narrowest index and widens it at the end.
        from torch._native.ops.sort import cutedsl_global

        x = torch.randn(130, nsort, device="cuda")
        with (
            mock.patch.object(
                cutedsl_global, "global_sort", wraps=cutedsl_global.global_sort
            ) as global_sort,
            self._functional_routed(),
        ):
            values, indices = torch.sort(x, stable=True)
        self.assertEqual(global_sort.call_args.args[2].dtype, narrow)
        ref_values, ref_indices = torch.sort(x, stable=True)
        self.assertEqual(values, ref_values)
        self.assertEqual(indices, ref_indices)

    def test_functional_layout_argsort_and_autograd(self):
        # Sorting a strided dim: outputs take stock's strides, argsort and
        # backward go through the same override.
        x = torch.randn(300, 4096, device="cuda").t().requires_grad_()
        result = []
        with self._functional_routed():
            launches = self._launches(
                lambda: result.extend(
                    (torch.sort(x, dim=0, stable=True), torch.argsort(x, dim=0))
                )
            )
        self.assertEqual(launches, 2)
        (values, indices), argsorted = result
        ref_values, ref_indices = torch.sort(x.detach(), dim=0, stable=True)
        self.assertEqual(values.stride(), ref_values.stride())
        self.assertEqual(indices.stride(), ref_indices.stride())
        self.assertEqual(values, ref_values)
        self.assertEqual(indices, ref_indices)
        self.assertEqual(x.detach().gather(0, argsorted), ref_values)
        values.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_indices_on_another_device_falls_back(self):
        x = torch.randn(4, 100, device="cuda")
        out = (torch.empty_like(x), torch.empty(4, 100, dtype=torch.uint16))
        self.assertEqual(
            self._launches(
                lambda: self.assertRaises(RuntimeError, torch.sort, x, out=out)
            ),
            0,
        )

    def test_row_length_beyond_int32_declines(self):
        # uint32 indices address rows the Int32 kernel shapes cannot.
        from torch._native.ops.sort import cutedsl_impl

        n = 2**31 + 16
        x = torch.zeros(1, 1, device="cuda").expand(1, n)
        out = {
            "values": torch.zeros(1, 1, device="cuda").expand(1, n),
            "indices": torch.zeros(1, 1, dtype=torch.uint32, device="cuda").expand(
                1, n
            ),
        }
        self.assertFalse(cutedsl_impl._out_cond(x, stable=True, dim=-1, **out))

    @parametrize("nsort", (1000, 5000))
    def test_outs_aliasing_input_fall_back(self, nsort):
        # Some leaves read x while writing values: aliasing goes to stock, which
        # copies first, or raises on a partial overlap.
        x = torch.randn(64, nsort, device="cuda")
        ref_values, ref_indices = torch.sort(x, stable=True)
        indices = torch.empty(x.shape, dtype=torch.int16, device="cuda")
        self.assertEqual(
            self._launches(lambda: torch.sort(x, stable=True, out=(x, indices))), 0
        )
        self.assertEqual(x, ref_values)
        self.assertEqual(indices.long(), ref_indices)
        buf = torch.randn(65 * nsort, device="cuda")
        x, values = buf[: 64 * nsort].view(64, nsort), buf[nsort:].view(64, nsort)
        with self.assertRaisesRegex(RuntimeError, "single memory location"):
            torch.sort(x, stable=True, out=(values, indices))

    def test_outs_aliasing_each_other_fall_back(self):
        # values and indices in one storage would race in the leaves.
        from torch._native.ops.sort import cutedsl_impl

        x = torch.randn(64, 1000, device="cuda")
        buf = torch.empty(2 * x.numel(), dtype=torch.int32, device="cuda")
        values, indices = (
            buf[: x.numel()].view(torch.float32).view(x.shape),
            buf[x.numel() :].view(x.shape),
        )
        self.assertFalse(
            cutedsl_impl._out_cond(
                x, stable=True, dim=-1, values=values, indices=indices
            )
        )
        self.assertEqual(
            self._launches(lambda: torch.sort(x, stable=True, out=(values, indices))), 0
        )
        ref_values, ref_indices = torch.sort(x, stable=True)
        self.assertEqual(values, ref_values)
        self.assertEqual(indices.long(), ref_indices)

    def test_resized_outs_take_input_strides(self):
        x = torch.randn(1000, 64, device="cuda").t()
        values = torch.empty(0, device="cuda")
        indices = torch.empty(0, dtype=torch.int16, device="cuda")
        self.assertEqual(
            self._launches(lambda: torch.sort(x, stable=True, out=(values, indices))), 1
        )
        ref = torch.sort(x, stable=True)
        self.assertEqual(values.stride(), ref.values.stride())
        self.assertEqual(indices.stride(), ref.values.stride())
        self.assertEqual(values, ref.values)
        self.assertEqual(indices.long(), ref.indices)

    def test_cow_outs_fire(self):
        x = torch.randn(64, 1000, device="cuda")
        source = torch.zeros(64, 1000, device="cuda")
        values = torch._lazy_clone(source)
        indices = torch._lazy_clone(
            torch.zeros(64, 1000, dtype=torch.int16, device="cuda")
        )
        self.assertEqual(
            self._launches(lambda: torch.sort(x, stable=True, out=(values, indices))), 1
        )
        self.assertEqual(values, torch.sort(x, stable=True).values)
        self.assertEqual(source, torch.zeros_like(source))

    def test_rows_beyond_int32_decline(self):
        # The kernels take the row count as Int32; stride-0 views avoid the memory.
        from torch._native.ops.sort import cutedsl_impl

        x = torch.zeros(1, 1, device="cuda").expand(2**31, 2)
        out = {
            "values": torch.zeros(1, 1, device="cuda").expand(2**31, 2),
            "indices": torch.zeros(1, 1, dtype=torch.uint8, device="cuda").expand(
                2**31, 2
            ),
        }
        self.assertFalse(cutedsl_impl._out_cond(x, stable=True, dim=-1, **out))
        self.assertTrue(
            cutedsl_impl._out_cond(
                x[: 2**31 - 1],
                stable=True,
                dim=-1,
                **{k: v[: 2**31 - 1] for k, v in out.items()},
            )
        )

    @parametrize("shape", ((50, 8000), (1, 70000), (300, 5000)))
    def test_wide_leaves_fire(self, shape):
        # Full-sort (< 128 rows), per-row (one row) and segmented leaves; float32
        # with int32 indices reuses the kernels test_sort_cutedsl_global compiles.
        x = torch.randn(*shape, device="cuda")
        indices = torch.empty(0, dtype=torch.int32, device="cuda")
        out = (torch.empty_like(x), indices)
        kernels = self._kernels(lambda: torch.sort(x, stable=True, out=out))
        self.assertEqual(kernels, {"global_sort": 1})
        self.assertEqual(indices.long(), torch.sort(x, stable=True).indices)


instantiate_parametrized_tests(TestCuTeDSLSortOverride)


if __name__ == "__main__":
    run_tests()
