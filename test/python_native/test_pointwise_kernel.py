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
    @parametrize("api", ("copy_", "_to_copy"))
    @parametrize(
        "source_dtype,target_dtype,value",
        (
            (torch.float64, torch.float16, 1 + 2**-11 + 2**-40),
            (torch.float64, torch.bfloat16, 1 + 2**-8 + 2**-40),
            (torch.int32, torch.bfloat16, 2**24 + 2**16 + 1),
            (torch.int64, torch.bfloat16, 2**24 + 2**16 + 1),
            (torch.uint32, torch.bfloat16, 2**24 + 2**16 + 1),
            (torch.uint64, torch.bfloat16, 2**24 + 2**16 + 1),
        ),
    )
    def test_conversion_matches_aten_rounding(
        self,
        device: str,
        api: str,
        source_dtype: torch.dtype,
        target_dtype: torch.dtype,
        value: float | int,
    ) -> None:
        from torch._native.ops.pointwise import overrides

        with torch.backends.python_native.cutedsl.disabled():
            source = torch.tensor([value], device=device, dtype=source_dtype).repeat(32)
            expected = source.to(target_dtype)
        actual = (
            overrides._to_copy_impl(source, dtype=target_dtype)
            if api == "_to_copy"
            else overrides._copy_impl(
                torch.empty_like(source, dtype=target_dtype), source
            )
        )
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_default_cast_rounds_through_common_dtype(self, device: str) -> None:
        import cutlass

        from torch._native.ops.pointwise import kernel, ops

        with torch.backends.python_native.cutedsl.disabled():
            x = torch.full((16,), 2049, dtype=torch.int32, device=device)
            y = torch.ones(16, dtype=torch.float16, device=device)
            expected = x + y
            (actual,) = kernel.run(
                ops.get_fn("_add"),
                ("default_common_dtype",),
                2,
                1,
                (cutlass.Float32(1),),
                cutlass.Float32,
                [x, y],
                (torch.float16,),
                load="direct",
            )
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize(
        "name,dtype",
        (
            ("_bitwise_right_shift", torch.uint8),
            ("_fmod_int", torch.uint8),
            ("_remainder_int", torch.uint8),
            ("_floor_divide_int", torch.int32),
            ("_floor_divide_int", torch.int64),
        ),
    )
    def test_integer_functor_boundaries(
        self, device: str, name: str, dtype: torch.dtype
    ) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        with torch.backends.python_native.cutedsl.disabled():
            if name == "_bitwise_right_shift":
                values, divisors = (200, 128), (8, 9)
                reference = torch.bitwise_right_shift
            elif name == "_floor_divide_int":
                values, divisors = (1, -1), (0, 0)
                reference = torch.floor_divide
            else:
                values, divisors = (200, 200), (7, 0 if name == "_remainder_int" else 3)
                reference = torch.fmod if name == "_fmod_int" else torch.remainder
            inputs = [
                torch.tensor(v, device=device, dtype=dtype) for v in (values, divisors)
            ]
            expected = reference(*inputs)
            (actual,) = kernel.run(
                ops.get_fn(name),
                ("integer_boundary", name, dtype),
                2,
                1,
                (),
                torch2cute[dtype],
                inputs,
                (dtype,),
                load="direct",
            )
            self.assertEqual(actual, expected)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("layout", ("dense", "rowgap", "strided"))
    def test_callable_without_name(self, device: str, layout: str) -> None:
        from functools import partial

        import cutlass

        from torch._native.ops.pointwise import kernel, ops

        with torch.backends.python_native.cutedsl.disabled():
            base = torch.arange(128, device=device, dtype=torch.float32).reshape(8, 16)
            x = base[:, ::2] if layout == "strided" else base[:, :8]
            if layout == "dense":
                x = x.contiguous()
            fn = partial(ops.get_fn("_mul"), y=2.0)
            (actual,) = kernel.run(
                fn,
                ("unnamed_callable",),
                1,
                1,
                (),
                cutlass.Float32,
                [x],
                (x.dtype,),
                load="direct",
            )
            self.assertEqual(actual, x * 2)

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
                load=None,
                out_refs=outputs,
            )

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("block,work_per_thread", ((256, 1), (320, 1), (256, 3)))
    def test_tuned_launch_index_boundary(
        self, device: str, block: int, work_per_thread: int
    ) -> None:
        import cutlass
        import cutlass.cute as cute

        from torch._native.cutedsl import launch
        from torch._native.ops.pointwise.kernel import _block_base

        count = 2**31 - 2
        tile = block * work_per_thread
        bid = (count - 1) // tile

        @cute.kernel
        def indices(out: Any, bid: Any) -> None:
            tid, _, _ = cute.arch.thread_idx()
            base = _block_base(bid, tid, cutlass.const_expr(tile))
            for p in cutlass.range_constexpr(work_per_thread):
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
            x = torch.full((3, 2), 1.5, device=device, dtype=torch.float32).t()
            expected = x + torch.arange(6, device=device).reshape(2, 3)
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
            load="direct",
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
                load="direct",
            )
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("work_per_thread", (1, 4))
    def test_rowgap_inputs_preserve_output_layout(
        self, device: str, work_per_thread: int
    ) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        count = None
        with torch.backends.python_native.cutedsl.disabled():
            for rows, cols, pitch in (
                (8, 8, 16),
                (12, 16, 20),
                (4, 12, 24),
                (5, 8, 9),
                (8, 8, 16),
            ):
                x = torch.arange(
                    rows * pitch, device=device, dtype=torch.float32
                ).reshape(rows, pitch)[:, :cols]
                expected = -x
                for transposed in (False, True, False):
                    out = torch.empty(
                        (cols, rows) if transposed else (rows, cols), device=device
                    )
                    if transposed:
                        out = out.t()
                    strides = out.stride()
                    (actual,) = kernel.run(
                        ops.get_fn("_neg"),
                        ("rowgap_output",),
                        1,
                        1,
                        (),
                        torch2cute[x.dtype],
                        [x],
                        (x.dtype,),
                        out=[out],
                        work_per_thread=work_per_thread,
                        load="direct",
                    )
                    self.assertIs(actual, out)
                    self.assertEqual(out.stride(), strides)
                    self.assertEqual(actual, expected, atol=0, rtol=0)
                if count is None:
                    count = len(kernel._KERNELS)
                self.assertEqual(len(kernel._KERNELS), count)

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
            load="direct",
        )
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @onlyCUDA
    @skipIfNoCuteDSL
    @dtypes(torch.float16, torch.float32, torch.float64)
    @parametrize("name", ("sin", "add"))
    def test_dense_presets_reuse_changing_shapes(
        self, device: str, dtype: torch.dtype, name: str
    ) -> None:
        from torch._native.cutedsl.dtypes import torch2cute
        from torch._native.ops.pointwise import kernel, ops

        ct = torch2cute[torch.float64 if dtype is torch.float64 else torch.float32]
        with torch.backends.python_native.cutedsl.disabled():
            before = len(kernel._KERNELS)
            for size in (64, 256, 8 * 1024**2, 4 * 1024**2, 1024, 0, 64):
                x = torch.randn(size, device=device, dtype=dtype)
                inputs = [x, x] if name == "add" else [x]
                expected = getattr(torch, name)(*inputs)
                (actual,) = kernel.run(
                    ops.get_fn("_" + name),
                    ("changing_shape",),
                    len(inputs),
                    1,
                    (ct(1),) if name == "add" else (),
                    ct,
                    inputs,
                    (dtype,),
                )
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual, expected)
                limit = 1 if name == "sin" and dtype is not torch.float16 else 2
                self.assertLessEqual(len(kernel._KERNELS) - before, limit)

    @onlyCUDA
    @skipIfNoCuteDSL
    @parametrize("cols", (1024, 1028, 4096))
    def test_rowgap_wide_preset_handles_vector_tails(
        self, device: str, cols: int
    ) -> None:
        import cutlass

        from torch._native.ops.pointwise import kernel, ops

        with torch.backends.python_native.cutedsl.disabled():
            rows = 4096 if cols == 4096 else 2048
            x, y = [
                torch.randn(rows, cols + 16, device=device, dtype=torch.float16)[
                    :, :cols
                ]
                for _ in range(2)
            ]
            expected = x + y
            (actual,) = kernel.run(
                ops.get_fn("_add"),
                ("rowgap_tail",),
                2,
                1,
                (cutlass.Float32(1),),
                cutlass.Float32,
                [x, y],
                (x.dtype,),
            )
            self.assertEqual(actual, expected)

    @onlyCUDA
    @skipIfNoCuteDSL
    def test_generic_kernel_reuses_collapsed_runtime_layouts(self, device: str) -> None:
        import cutlass

        from torch._native.ops.pointwise import kernel, ops

        def view(shape: tuple[int, ...], strides: tuple[int, ...]) -> torch.Tensor:
            span = sum((n - 1) * s for n, s in zip(shape, strides)) + 1
            return torch.randn(span + 2, device=device).as_strided(shape, strides, 1)

        cases = (
            ((7,), (2,), (1,), (1,), (4,)),
            ((5, 7), (19, 2), (1, 7), (0, 3), (1, 8)),
            ((2, 3, 5), (55, 17, 2), (2, 1, 5), (23, 0, 3), (49, 1, 7)),
            ((), (), (), (), ()),
            ((1,) * 30 + (3, 5), (0,) * 30 + (13, 2), (1,), (1,), (0,) * 30 + (17, 3)),
        )
        count = None
        with torch.backends.python_native.cutedsl.disabled():
            for shape, strides, rhs_shape, rhs_strides, out_strides in cases * 2:
                x, y, out = (
                    view(shape, strides),
                    view(rhs_shape, rhs_strides),
                    view(shape, out_strides),
                )
                expected = x + y
                (actual,) = kernel.run(
                    ops.get_fn("_add"),
                    ("runtime_layouts",),
                    2,
                    1,
                    (cutlass.Float32(1),),
                    cutlass.Float32,
                    [x, y],
                    (torch.float32,),
                    out=[out],
                    load="direct",
                )
                self.assertIs(actual, out)
                self.assertEqual(actual, expected)
                if count is None:
                    count = len(kernel._KERNELS)
                self.assertEqual(len(kernel._KERNELS), count)

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
            load="direct",
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
            load="direct",
        )
        with torch.backends.python_native.cutedsl.disabled():
            self.assertEqual(actual, expected)


instantiate_device_type_tests(TestPointwiseKernel, globals(), only_for=("cuda",))

if __name__ == "__main__":
    run_tests()
