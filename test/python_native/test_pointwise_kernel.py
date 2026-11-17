# Owner(s): ["module: dsl-native-ops"]

from __future__ import annotations

from typing import Annotated, Any

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


@instantiate_parametrized_tests
class TestPointwiseIndexing(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "shape,strides,expected",
        (((2**31,), (1,), False), ((2,), (2**31,), False), ((8, 8), (16, 1), True)),
    )
    def test_index_width_checks_count_and_storage_span(
        self, shape: tuple[int, ...], strides: tuple[int, ...], expected: bool
    ) -> None:
        from torch._native.utils.tensor import can_use_32bit_indexing

        value = torch.empty_strided(shape, strides, device="meta")
        self.assertEqual(can_use_32bit_indexing(value), expected)


class TestPointwiseKernel(TestCase):
    hw_classification = HardwareClassification.CUDA

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_supplied_output_index_span(self, device: str) -> None:
        import cutlass

        from torch._native.ops.pointwise import kernel, ops
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            source = torch.empty(5, device=device)
            outputs = (
                torch.empty_like(source),
                torch.empty_strided((5,), (2**30,), device=device, dtype=torch.int32),
            )
        with self.assertRaisesRegex(ValueError, "32-bit counts and element offsets"):
            kernel._build_plan(
                ops.get_fn("_frexp"),
                "output_index_span",
                1,
                2,
                (),
                cutlass.Float32,
                [source],
                (torch.float32, torch.int32),
                None,
                None,
                None,
                out_refs=outputs,
            )

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("block,ept", ((256, 1), (320, 1), (256, 3)))
    def test_tuned_launch_index_boundary(
        self, device: str, block: int, ept: int
    ) -> None:
        import cutlass
        import cutlass.cute as cute

        from torch._native.cutedsl import launch
        from torch._native.ops.pointwise.kernel import _block_base

        count = 2**31 - 2
        tile = block * ept
        bid = (count - 1) // tile

        @cute.kernel
        def indices(out: Any, bid: Any) -> None:
            tid, _, _ = cute.arch.thread_idx()
            base = _block_base(bid, tid, cutlass.const_expr(tile))
            for p in cutlass.range_constexpr(ept):
                i = base + p * block
                out[tid + p * block] = cutlass.Int64(i)

        @cute.jit
        def run(out: Any, bid: Any, stream: Any) -> None:
            indices(out, bid).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

        actual = torch.empty(tile, device=device, dtype=torch.int64)
        fake = launch.fake_compact(cutlass.Int64, (tile,), align=16)
        fn = launch.compile_kernel(run, fake, cutlass.Int32(1), launch.stream())
        fn(actual, cutlass.Int32(bid), launch.stream())
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.arange(
                bid * tile, (bid + 1) * tile, device=device, dtype=torch.int64
            )
        self.assertEqual(actual, expected)
        self.assertEqual(actual < count, expected < count)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("strided", (False, True))
    @parametrize("precision", ("storage", "compute", "float64", "compute_to_float64"))
    def test_signature_selected_casts(
        self, device: str, strided: bool, precision: str
    ) -> None:
        import cutlass.cute as cute

        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel
        from torch._native.ops.pointwise.signature import specialize

        @cute.jit
        def narrow(x: Annotated[Any, "storage"]) -> Annotated[Any, "storage"]:
            return (x + x.__class__(0.0001)) - x

        @cute.jit
        def opmath(x: Annotated[Any, "compute"]) -> Annotated[Any, "compute"]:
            return (x + x.__class__(0.0001)) - x

        @cute.jit
        def wide(x: Annotated[Any, torch.float64]) -> Annotated[Any, torch.float64]:
            return (x + x.__class__(0.0001)) - x

        @cute.jit
        def opmath_wide(x: Annotated[Any, "compute"]) -> Annotated[Any, torch.float64]:
            return (x + x.__class__(0.0001)) - x

        fn = {
            "storage": narrow,
            "compute": opmath,
            "float64": wide,
            "compute_to_float64": opmath_wide,
        }[precision]
        sig = specialize(fn, torch.float16, torch.float16)
        with torch.backends.python_native.cutedsl.disabled():
            x = torch.ones(64, device=device, dtype=torch.float16)
            x = x[::2] if strided else x
            dtype = {
                "storage": torch.float16,
                "compute": torch.float32,
                "float64": torch.float64,
                "compute_to_float64": torch.float32,
            }[precision]
            result = torch.float64 if "float64" in precision else torch.float16
            work = x.to(dtype)
            expected = ((work + 0.0001) - work).to(result).to(torch.float32)
        (actual,) = kernel.run(
            fn,
            ("signature_casts", precision, strided),
            1,
            1,
            (),
            torch2cute[sig.compute],
            [x],
            (torch.float32,),
            input_casts=(sig.input_dtype(0, False),),
            result_dtypes=sig.results,
        )
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_indexed_tensor_preserves_fraction(self, device: str) -> None:
        import cutlass.cute as cute

        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel

        @cute.jit
        def indexed(
            i: Annotated[Any, torch.int64], x: Annotated[Any, "storage"]
        ) -> Annotated[Any, "storage"]:
            return x + x.__class__(i)

        with torch.backends.python_native.cutedsl.disabled():
            x = torch.full((4,), 1.5, device=device, dtype=torch.float32)
            expected = x + torch.arange(4, device=device)
        (actual,) = kernel.run(
            indexed,
            ("indexed_input",),
            1,
            1,
            (),
            torch2cute[x.dtype],
            [x],
            (x.dtype,),
            with_index=True,
        )
        self.assertEqual(actual, expected)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("strided", (False, True))
    def test_signature_mixed_returns(self, device: str, strided: bool) -> None:
        import cutlass
        import cutlass.cute as cute

        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel
        from torch._native.ops.pointwise.signature import specialize

        @cute.jit
        def split(
            x: Annotated[Any, "storage"],
        ) -> tuple[Annotated[Any, torch.bool], Annotated[Any, torch.int32]]:
            return x > x.__class__(0), cutlass.Int32(x)

        sig = specialize(split, torch.float16, torch.float16, 2)
        with torch.backends.python_native.cutedsl.disabled():
            x = torch.arange(-32, 32, device=device, dtype=torch.float16)
            x = x[::2] if strided else x
            expected = (x > 0, x.to(torch.int32))
        actual = kernel.run(
            split,
            ("signature_returns", strided),
            1,
            2,
            (),
            torch2cute[sig.compute],
            [x],
            sig.results,
            input_casts=(sig.input_dtype(0, False),),
            result_dtypes=sig.results,
        )
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("strided", (False, True))
    def test_signbit_all_storage_patterns(
        self, device: str, dtype: torch.dtype, strided: bool
    ) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops
        from torch._native.ops.pointwise.signature import Functor
        from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind

        with torch.backends.python_native.cutedsl.disabled():
            bits = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
            values = (
                bits.repeat_interleave(2).view(dtype)[::2]
                if strided
                else bits.view(dtype)
            )
            expected = torch.signbit(values)
        sig = Functor("_signbit").signature(
            ((dtype, False),), PromotionKind.ALWAYS_BOOL
        )
        (actual,) = kernel.run(
            ops.get_fn("_signbit"),
            ("signbit_patterns", dtype, strided),
            1,
            1,
            (),
            torch2cute[sig.compute],
            [values],
            (torch.bool,),
            input_casts=(dtype,),
            result_dtypes=sig.results,
        )
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_low_level_launcher_rejects_another_current_device(
        self, device: str
    ) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two CUDA devices")
        with torch.cuda.device(0):
            x = torch.ones(4096, device="cuda:1")
            with self.assertRaisesRegex(
                AssertionError, "input on cuda:1, but cuda:0 is current"
            ):
                from cutlass import Float32

                from torch._native.ops.pointwise import kernel, ops

                kernel.run(
                    ops.get_fn("_identity"), (), 1, 1, (), Float32, [x], (x.dtype,)
                )

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize(
        "storage,compute,vec_bits",
        (
            (torch.float64, torch.float32, 128),
            (torch.float16, torch.float64, 64),
            (torch.float16, torch.float64, 256),
        ),
    )
    def test_storage_and_compute_widths(
        self, device: str, storage: torch.dtype, compute: torch.dtype, vec_bits: int
    ) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        value = torch.arange(64, device=device, dtype=storage)
        with torch.backends.python_native.cutedsl.disabled():
            expected = value.to(compute)
        (actual,) = kernel.run(
            ops.get_fn("_identity"),
            ("copy_width", storage, compute, vec_bits),
            1,
            1,
            (),
            torch2cute[compute],
            (value,),
            (compute,),
            vec_bits=vec_bits,
        )
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("change", ("function", "output", "storage", "compute"))
    def test_fixed_key_preserves_specialization(self, device: str, change: str) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        for second in (False, True, False):
            wide = change in ("storage", "compute")
            storage = torch.float64 if wide else torch.float32
            compute = torch.float64 if wide else torch.float32
            dtype = compute
            name = "_neg" if change == "function" and second else "_identity"
            if change == "output" and second:
                dtype = torch.float64
            elif change == "storage" and not second:
                storage = torch.float32
            elif change == "compute" and not second:
                compute = torch.float32
            with torch.backends.python_native.cutedsl.disabled():
                x = torch.full((64,), 1.00000001, device=device, dtype=storage)
                expected = x.to(compute)
                if name == "_neg":
                    expected = -expected
                expected = expected.to(dtype)
            (actual,) = kernel.run(
                ops.get_fn(name),
                ("specialization", change),
                1,
                1,
                (),
                torch2cute[compute],
                [x],
                (dtype,),
            )
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("ept", (1, 4))
    def test_rowgap_inputs_preserve_output_layout(self, device: str, ept: int) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        with torch.backends.python_native.cutedsl.disabled():
            x = torch.arange(128, device=device, dtype=torch.float32).reshape(8, 16)[
                :, :8
            ]
            expected = -x
        for transposed in (False, True, False):
            out = torch.empty((8, 8), device=device)
            if transposed:
                out = out.t()
            strides = out.stride()
            (actual,) = kernel.run(
                ops.get_fn("_neg"),
                ("rowgap_output", tuple(x.shape)),
                1,
                1,
                (),
                torch2cute[x.dtype],
                [x],
                (x.dtype,),
                out=[out],
                ept=ept,
            )
            self.assertIs(actual, out)
            self.assertEqual(out.stride(), strides)
            with torch.backends.python_native.cutedsl.disabled():
                self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("kind", ("device", "host"))
    def test_scalar_vector_rounding_and_provenance(
        self, device: str, kind: str
    ) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        x = torch.full((32,), -10000.0, device=device, dtype=torch.float16)
        y = torch.tensor(10000.25, device=device) if kind == "device" else 10000.25
        with torch.backends.python_native.cutedsl.disabled():
            expected = x + y
        scalar = (
            y
            if isinstance(y, torch.Tensor)
            else torch.tensor(y, dtype=torch.float64, device="cpu")
        )
        ct = torch2cute[torch.float32]
        (actual,) = kernel.run(
            ops.get_fn("_add"),
            ("scalar_precision", kind),
            2,
            1,
            (ct(1),),
            ct,
            [x, scalar],
            (x.dtype,),
        )
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_fixed_key_preserves_changing_shapes(self, device: str) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        for size in (64, 256, 1024, 128, 32, 0, 64):
            x = torch.arange(size, device=device, dtype=torch.float32)
            with torch.backends.python_native.cutedsl.disabled():
                expected = -x
            (actual,) = kernel.run(
                ops.get_fn("_neg"),
                ("changing_shape",),
                1,
                1,
                (),
                torch2cute[x.dtype],
                [x],
                (x.dtype,),
            )
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("name", ("maximum", "minimum", "fmax", "fmin"))
    def test_extrema_preserve_signed_zeros(self, device: str, name: str) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        with torch.backends.python_native.cutedsl.disabled():
            a = torch.tensor(
                [-0.0, -0.0, 0.0, 0.0, float("nan"), 1.0, float("nan"), float("inf")],
                device=device,
                dtype=torch.float64,
            )
            b = torch.tensor(
                [-0.0, 0.0, -0.0, 0.0, 1.0, float("nan"), float("nan"), float("inf")],
                device=device,
                dtype=torch.float64,
            )
            expected = getattr(torch, name)(a, b)
        (actual,) = kernel.run(
            ops.get_fn(f"_{name}"),
            ("signed_zero", name),
            2,
            1,
            (),
            torch2cute[a.dtype],
            [a, b],
            (a.dtype,),
        )
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)
            zeros = expected == 0
            self.assertEqual(
                torch.signbit(actual[zeros]), torch.signbit(expected[zeros])
            )

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("dtype", (torch.int32, torch.int64))
    @parametrize("direction", ("left", "right"))
    @parametrize("layout", ("dense", "broadcast"))
    def test_shift_count_bounds(
        self, device: str, dtype: torch.dtype, direction: str, layout: str
    ) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        counts = [-1, 0, 31, 32, 33, 63, 64, 65]
        if dtype is torch.int64:
            counts += [2**32, 2**32 + 1, -(2**32), -(2**32) + 1]
        y = torch.tensor(counts, device=device, dtype=dtype)
        x = torch.tensor([1, -1], device=device, dtype=dtype).view(2, 1)
        if layout == "dense":
            x, y = x.expand(2, len(counts)).contiguous(), y.repeat(2, 1)
        else:
            y = y.view(1, -1)
        name = f"bitwise_{direction}_shift"
        with torch.backends.python_native.cutedsl.disabled():
            expected = getattr(torch, name)(x, y)
        (actual,) = kernel.run(
            ops.get_fn(f"_{name}"),
            ("shift_bounds", dtype, direction, layout),
            2,
            1,
            (),
            torch2cute[dtype],
            [x, y],
            (dtype,),
        )
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestPointwiseKernel, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
