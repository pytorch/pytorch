# Owner(s): ["module: linear algebra"]

import inspect

import torch
from torch.nn import functional as F
from torch.nn.functional import SwizzleType
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import (
    _f32_to_e8m0_rceil,
    compute_error,
    from_blocked,
    from_blocked_format,
    mxfp8_32x32_swizzle_f,
    to_mxfp as to_mxfp8_reference,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    subtest,
    TestCase,
)
from torch.testing._internal.mxfp8_test_utils import (
    assert_mxfp8_semantics,
    make_f32_to_e8m0_rceil_cases,
    make_mxfp8_semantic_cases,
)


def _quantize_mxfp8_reference(
    input: torch.Tensor,
    *,
    qdata_dtype: torch.dtype,
    scaling_algorithm: F.ScalingAlgorithm,
    scaling_type: F.ScalingType,
    swizzle_type: SwizzleType = SwizzleType.NO_SWIZZLE,
):
    if (
        qdata_dtype != torch.float8_e4m3fn
        or scaling_algorithm != F.ScalingAlgorithm.MXFP_E8M0_RU
        or scaling_type != F.ScalingType.BlockWise1x32
    ):
        raise ValueError("unsupported MXFP8 reference recipe")
    scales, qdata = to_mxfp8_reference(input, format="mxfp8", swizzle_type=swizzle_type)
    return qdata, scales


_NVIDIA_SM100_OR_LATER = torch.version.rocm is None and bool(SM100OrLater)


_MXFP8_IMPLEMENTATIONS = (
    subtest(_quantize_mxfp8_reference, name="reference"),
    subtest(F.quantize_tensor, name="public", decorators=[skipIfNoCuteDSL]),
)
_MXFP8_KWARGS = {
    "qdata_dtype": torch.float8_e4m3fn,
    "scaling_algorithm": F.ScalingAlgorithm.MXFP_E8M0_RU,
    "scaling_type": F.ScalingType.BlockWise1x32,
}
_MXFP8_NO_SWIZZLE_KWARGS = {
    **_MXFP8_KWARGS,
    "swizzle_type": SwizzleType.NO_SWIZZLE,
}


class TestMXFP8ReferenceNumerics(TestCase):
    def test_f32_to_e8m0_rceil(self, device):
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L113
        values, expected = make_f32_to_e8m0_rceil_cases(device=device)
        self.assertEqual(_f32_to_e8m0_rceil(values), expected.to(device))

    @parametrize("quantize_fn", _MXFP8_IMPLEMENTATIONS)
    @parametrize("input_dtype", (torch.float32, torch.bfloat16))
    def test_mxfp8_corner_case_bytes(self, quantize_fn, input_dtype, device):
        if quantize_fn is F.quantize_tensor and not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L264
        cases = make_mxfp8_semantic_cases(input_dtype, device=device)
        qdata, scales = quantize_fn(cases.inputs, **_MXFP8_NO_SWIZZLE_KWARGS)
        assert_mxfp8_semantics(qdata, scales, cases)

    @parametrize(
        "swizzle_type",
        (SwizzleType.NO_SWIZZLE, SwizzleType.SWIZZLE_32_4_4),
        name_fn=lambda swizzle_type: swizzle_type.name,
    )
    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        (
            (1, 32),
            (31, 96),
            (128, 64),
            (128, 128),
            (129, 160),
            (256, 256),
            (1024, 1152),
            (2080, 2048),
            (2048, 4224),
        ),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_randn_sqnr(self, swizzle_type, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        swizzled = swizzle_type == SwizzleType.SWIZZLE_32_4_4
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp, **_MXFP8_KWARGS, swizzle_type=swizzle_type
        )
        qdata, scales = F.quantize_tensor(
            data_hp, **_MXFP8_KWARGS, swizzle_type=swizzle_type
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(qdata.shape, data_hp.shape)
        expected_scale_shape = (
            ((shape[0] + 127) // 128, (shape[1] + 127) // 128, 32, 16)
            if swizzled
            else (shape[0], shape[1] // 32)
        )
        self.assertEqual(tuple(scales.shape), expected_scale_shape)
        if swizzled:
            scale_bytes = scales.view(torch.uint8).flatten()
            self.assertEqual(scale_bytes, scales_ref.view(torch.uint8))
            scales = from_blocked(qdata, scales.flatten(), 32)
        else:
            self.assertEqual(scales.view(torch.uint8), scales_ref.view(torch.uint8))
        dequantized = from_blocked_format(qdata, scales).float()
        sqnr = compute_error(data_hp.float(), dequantized)
        self.assertGreater(sqnr.item(), 18.0)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 16), (64, 128), (160, 144), (2048, 2080)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_m(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp.t().contiguous(),
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        qdata, scales = F.quantize_tensor(
            data_hp.t(), **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(tuple(qdata.shape), (shape[1], shape[0]))
        self.assertEqual(
            tuple(scales.shape),
            ((shape[1] + 127) // 128, (shape[0] + 127) // 128, 32, 16),
        )
        scale_bytes = scales.view(torch.uint8).flatten()
        self.assertEqual(scale_bytes, scales_ref.view(torch.uint8))

    @parametrize("input_dtype", (torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32896, 128), (11008, 384)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_m_large_k_tile_boundary(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(*shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = _quantize_mxfp8_reference(
            data_hp.t().contiguous(),
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        qdata, scales = F.quantize_tensor(
            data_hp.t(), **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(
            scales.view(torch.uint8).flatten(), scales_ref.view(torch.uint8)
        )

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 32), (64, 128), (160, 160), (2048, 2080)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_km(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_k_ref, scales_k_ref = _quantize_mxfp8_reference(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        qdata_m_ref, scales_m_ref = _quantize_mxfp8_reference(
            data_hp.t().contiguous(),
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
        )
        qdata_k, scales_k, qdata_m, scales_m = F.quantize_tensor_dual(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(qdata_k.view(torch.uint8), qdata_k_ref.view(torch.uint8))
        self.assertEqual(qdata_m.view(torch.uint8), qdata_m_ref.view(torch.uint8))
        scale_k_bytes = scales_k.view(torch.uint8).flatten()
        scale_m_bytes = scales_m.view(torch.uint8).flatten()
        self.assertEqual(scale_k_bytes, scales_k_ref.view(torch.uint8))
        self.assertEqual(scale_m_bytes, scales_m_ref.view(torch.uint8))

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @parametrize(
        "shape",
        ((32, 32), (64, 128), (160, 160), (2048, 4224)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_k_square(self, input_dtype, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(shape, device=device, dtype=input_dtype)
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp)
        qdata, scales = F.quantize_tensor(
            data_hp,
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_use_square_block_size=True,
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(
            tuple(scales.shape),
            ((shape[0] + 127) // 128, (shape[1] + 127) // 128, 32, 16),
        )
        self.assertEqual(
            scales.view(torch.uint8).flatten(), scales_ref.view(torch.uint8)
        )

    @parametrize("input_dtype", (torch.float32, torch.bfloat16, torch.float16))
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_k_square_nan(self, input_dtype, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.randn(64, 64, device=device, dtype=input_dtype)
        data_hp[0, 0] = float("nan")
        qdata_ref, scales_ref = mxfp8_32x32_swizzle_f(data_hp)
        qdata, scales = F.quantize_tensor(
            data_hp,
            **_MXFP8_KWARGS,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            scaling_type_use_square_block_size=True,
        )
        self.assertEqual(qdata.view(torch.uint8), qdata_ref.view(torch.uint8))
        self.assertEqual(
            scales.view(torch.uint8).flatten(), scales_ref.view(torch.uint8)
        )
        expected_nan = torch.zeros((64, 64), device=device, dtype=torch.bool)
        expected_nan[:32, :32] = True
        self.assertEqual(torch.isnan(qdata), expected_nan)

        scale_bytes = from_blocked(qdata, scales.flatten(), 32).view(torch.uint8)
        expected_nan_scale = torch.zeros((64, 2), device=device, dtype=torch.bool)
        expected_nan_scale[:32, 0] = True
        self.assertEqual(scale_bytes == 255, expected_nan_scale)

    @parametrize(
        "quant_orientation,is_square_scaling,is_scale_swizzled",
        (
            subtest(("dim_k", False, False), name="dim_k_compact"),
            subtest(("dim_k", False, True), name="dim_k_swizzled"),
            subtest(("dim_k", True, True), name="dim_k_square"),
            subtest(("dim_m", False, True), name="dim_m"),
        ),
    )
    @parametrize(
        "shape",
        ((0, 32), (32, 0), (0, 0)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_empty(
        self, quant_orientation, is_square_scaling, is_scale_swizzled, shape, device
    ):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        M, K = shape
        scale_k_shape = (
            ((M + 127) // 128, (K + 127) // 128, 32, 16)
            if is_scale_swizzled
            else (M, K // 32)
        )
        scale_m_shape = ((K + 127) // 128, (M + 127) // 128, 32, 16)
        expected_shapes = {
            "dim_k": ((M, K), scale_k_shape),
            "dim_m": ((K, M), scale_m_shape),
        }[quant_orientation]
        api_input = data_hp.t() if quant_orientation == "dim_m" else data_hp
        swizzle_type = (
            SwizzleType.SWIZZLE_32_4_4 if is_scale_swizzled else SwizzleType.NO_SWIZZLE
        )
        outputs = F.quantize_tensor(
            api_input,
            **_MXFP8_KWARGS,
            swizzle_type=swizzle_type,
            scaling_type_use_square_block_size=is_square_scaling,
        )
        self.assertEqual(len(outputs), len(expected_shapes))
        for index, (output, expected_shape) in enumerate(
            zip(outputs, expected_shapes, strict=True)
        ):
            self.assertEqual(tuple(output.shape), expected_shape)
            expected_dtype = (
                torch.float8_e4m3fn if index % 2 == 0 else torch.float8_e8m0fnu
            )
            self.assertEqual(output.dtype, expected_dtype)
            self.assertEqual(output.device, data_hp.device)
            self.assertEqual(output.numel(), 0)

    @parametrize(
        "shape",
        ((0, 32), (32, 0), (0, 0)),
        name_fn=lambda shape: f"M{shape[0]}_K{shape[1]}",
    )
    @skipIfNoCuteDSL
    def test_to_mx_rceil_dim_km_empty(self, shape, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        M, K = shape
        data_hp = torch.empty(shape, dtype=torch.bfloat16, device=device)
        outputs = F.quantize_tensor_dual(
            data_hp, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        expected_shapes = (
            (M, K),
            ((M + 127) // 128, (K + 127) // 128, 32, 16),
            (K, M),
            ((K + 127) // 128, (M + 127) // 128, 32, 16),
        )
        for index, (output, expected_shape) in enumerate(
            zip(outputs, expected_shapes, strict=True)
        ):
            self.assertEqual(tuple(output.shape), expected_shape)
            self.assertEqual(
                output.dtype,
                torch.float8_e4m3fn if index % 2 == 0 else torch.float8_e8m0fnu,
            )
            self.assertEqual(output.numel(), 0)

    @parametrize("transposed,square", ((False, False), (True, False), (False, True)))
    @skipIfNoCuteDSL
    def test_quantize_tensor_compile(self, transposed, square, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        data = torch.randn((160, 160), dtype=torch.bfloat16, device=device)
        api_input = data.t() if transposed else data

        def fn(x):
            return F.quantize_tensor(
                x,
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
                scaling_type_use_square_block_size=square,
            )

        expected = fn(api_input)
        actual = torch.compile(fn, fullgraph=True)(api_input)
        for got, reference in zip(actual, expected, strict=True):
            self.assertEqual(got.view(torch.uint8), reference.view(torch.uint8))

    @skipIfNoCuteDSL
    def test_quantize_tensor_scaled_mm(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        mat_a = torch.randn((128, 256), dtype=torch.bfloat16, device=device)
        mat_b = torch.randn((256, 192), dtype=torch.bfloat16, device=device)
        swizzle = SwizzleType.SWIZZLE_32_4_4
        qa, sa = F.quantize_tensor(mat_a, **_MXFP8_KWARGS, swizzle_type=swizzle)
        qb, sb = F.quantize_tensor(mat_b.t(), **_MXFP8_KWARGS, swizzle_type=swizzle)
        out = F.scaled_mm(
            qa,
            qb.t(),
            sa,
            F.ScalingType.BlockWise1x32,
            sb,
            F.ScalingType.BlockWise1x32,
            swizzle_a=swizzle,
            swizzle_b=swizzle,
        )
        self.assertEqual(tuple(out.shape), (128, 192))
        self.assertGreater(
            compute_error(mat_a.float() @ mat_b.float(), out.float()).item(), 15.0
        )

    @skipIfNoCuteDSL
    def test_quantize_tensor_dual_compile(self, device):
        if not _NVIDIA_SM100_OR_LATER:
            self.skipTest("MXFP8 TMA requires NVIDIA SM100 or newer")
        input = torch.randn((160, 192), dtype=torch.bfloat16, device=device)

        def fn(x):
            return F.quantize_tensor_dual(
                x, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
            )

        expected = fn(input)
        actual = torch.compile(fn, fullgraph=True)(input)
        for got, reference in zip(actual, expected, strict=True):
            self.assertEqual(got.view(torch.uint8), reference.view(torch.uint8))


@instantiate_parametrized_tests
class TestQuantizeTensorMeta(TestCase):
    @parametrize(
        "quantize_fn,kwargs",
        (
            subtest((F.quantize_tensor, _MXFP8_NO_SWIZZLE_KWARGS), name="single"),
            subtest(
                (
                    F.quantize_tensor_dual,
                    {**_MXFP8_KWARGS, "swizzle_type": SwizzleType.SWIZZLE_32_4_4},
                ),
                name="dual",
            ),
        ),
    )
    def test_meta_validates_arguments(self, quantize_fn, kwargs):
        input = torch.empty((32, 32), dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(ValueError, "MXFP_E8M0_RU"):
            quantize_fn(input, **(kwargs | {"scaling_algorithm": 1}))
        with self.assertRaisesRegex(ValueError, "2D"):
            quantize_fn(input.flatten(), **kwargs)
        input.requires_grad_()
        outputs = quantize_fn(input, **kwargs)
        self.assertTrue(all(output.requires_grad for output in outputs))

    def test_quantize_tensor_argument_names(self):
        self.assertEqual(F.ScalingAlgorithm.MXFP_E8M0_RU.value, 0)
        public_names = tuple(inspect.signature(F.quantize_tensor).parameters)
        dual_names = tuple(inspect.signature(F.quantize_tensor_dual).parameters)
        native = torch.ops.aten._quantize_tensor.default._schema.arguments
        native_dual = torch.ops.aten._quantize_tensor_dual.default._schema.arguments
        self.assertEqual(
            public_names,
            (
                "input",
                "scaling_type",
                "qdata_dtype",
                "scaling_algorithm",
                "swizzle_type",
                "scaling_type_use_square_block_size",
            ),
        )
        self.assertEqual(public_names, dual_names)
        native_names = tuple(arg.name for arg in native)
        self.assertEqual(native_names, tuple(arg.name for arg in native_dual))
        self.assertEqual(public_names, native_names)
        for fn in (F.quantize_tensor, F.quantize_tensor_dual):
            params = tuple(inspect.signature(fn).parameters.values())
            self.assertEqual(params[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for param in params[1:]:
                self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(params[4].default, inspect.Parameter.empty)
        for schema_args in (native, native_dual):
            self.assertEqual(str(schema_args[3].type), "int")
            self.assertFalse(schema_args[0].kwarg_only)
            for arg in schema_args[1:]:
                self.assertTrue(arg.kwarg_only)
            self.assertFalse(schema_args[4].has_default_value())

    @parametrize(
        "shape,transposed,square,swizzle",
        (
            ((160, 160), False, False, SwizzleType.NO_SWIZZLE),
            ((160, 160), False, False, SwizzleType.SWIZZLE_32_4_4),
            ((160, 160), True, False, SwizzleType.SWIZZLE_32_4_4),
            ((160, 160), False, True, SwizzleType.SWIZZLE_32_4_4),
            ((0, 32), False, False, SwizzleType.NO_SWIZZLE),
        ),
    )
    def test_meta_shapes(self, shape, transposed, square, swizzle):
        source = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        api_input = source.t() if transposed else source
        qdata, scales = F.quantize_tensor(
            api_input,
            **_MXFP8_KWARGS,
            swizzle_type=swizzle,
            scaling_type_use_square_block_size=square,
        )
        rows, cols = api_input.shape
        expected_scale_shape = (
            ((rows + 127) // 128, (cols + 127) // 128, 32, 16)
            if swizzle == SwizzleType.SWIZZLE_32_4_4
            else (rows, cols // 32)
        )
        self.assertEqual(tuple(qdata.shape), (rows, cols))
        self.assertEqual(tuple(scales.shape), expected_scale_shape)
        self.assertEqual(qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float8_e8m0fnu)

    def test_fake_cuda_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        if not torch.backends.cuda.is_built():
            self.skipTest("requires a CUDA build")
        with FakeTensorMode():
            source = torch.empty((160, 160), dtype=torch.float16, device="cuda")
            qdata, scales = F.quantize_tensor(
                source.t(),
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            )
            self.assertEqual(tuple(qdata.shape), (160, 160))
            self.assertEqual(tuple(scales.shape), (2, 2, 32, 16))
            dual_input = torch.empty((160, 288), dtype=torch.float16, device="cuda")
            qk, sk, qm, sm = F.quantize_tensor_dual(
                dual_input,
                **_MXFP8_KWARGS,
                swizzle_type=SwizzleType.SWIZZLE_32_4_4,
            )
            self.assertEqual(tuple(qk.shape), (160, 288))
            self.assertEqual(tuple(sk.shape), (2, 3, 32, 16))
            self.assertEqual(tuple(qm.shape), (288, 160))
            self.assertEqual(tuple(sm.shape), (3, 2, 32, 16))

    def test_dispatch_transposed_shape(self):
        input = torch.empty((32, 160), dtype=torch.float16, device="meta").t()
        qdata, scales = torch._quantize_tensor(  # pyrefly: ignore[missing-attribute]
            input,
            qdata_dtype=torch.float8_e4m3fn,
            scaling_algorithm=F.ScalingAlgorithm.MXFP_E8M0_RU.value,
            scaling_type=F.ScalingType.BlockWise1x32.value,
            swizzle_type=SwizzleType.SWIZZLE_32_4_4.value,
        )
        self.assertEqual(tuple(qdata.shape), (160, 32))
        self.assertEqual(tuple(scales.shape), (2, 1, 32, 16))

    def test_dual_meta_shapes(self):
        input = torch.empty((160, 288), dtype=torch.bfloat16, device="meta")
        outputs = F.quantize_tensor_dual(
            input, **_MXFP8_KWARGS, swizzle_type=SwizzleType.SWIZZLE_32_4_4
        )
        self.assertEqual(
            tuple(tuple(output.shape) for output in outputs),
            ((160, 288), (2, 3, 32, 16), (288, 160), (3, 2, 32, 16)),
        )
        self.assertEqual(
            tuple(output.dtype for output in outputs),
            (
                torch.float8_e4m3fn,
                torch.float8_e8m0fnu,
                torch.float8_e4m3fn,
                torch.float8_e8m0fnu,
            ),
        )


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
