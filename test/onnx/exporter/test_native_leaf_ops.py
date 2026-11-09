# Owner(s): ["module: onnx"]
"""Numerical and registration coverage for native arithmetic translations."""

from __future__ import annotations

import math
import operator
from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from test_native_torchlib import no_script

import torch
from torch.onnx._internal.exporter import (
    _core,
    _custom_translation,
    _ir_builder,
    _registration,
)
from torch.onnx._internal.exporter._torchlib.ops import common, core
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


aten = torch.ops.aten
prims = torch.ops.prims

# Public source aliases, with real and complex registrations accounted separately.
_EXPECTED_CORE_TARGETS = (
    ("aten_scalar_tensor", False, (aten.scalar_tensor.default,)),
    ("aten_scalar_tensor_complex", True, (aten.scalar_tensor.default,)),
    ("aten_abs", False, (aten.abs.default, operator.abs)),
    ("aten_abs_complex", True, (aten.abs.default,)),
    ("aten__local_scalar_dense", False, (aten._local_scalar_dense.default,)),
    ("aten_acos", False, (aten.acos.default,)),
    ("aten_acosh", False, (aten.acosh.default,)),
    ("aten_add", False, (aten.add.Tensor,)),
    ("aten_add_scalar", False, (aten.add.Scalar,)),
    ("operator_add", False, (operator.add,)),
    ("aten_add_complex", True, (aten.add.Tensor, aten.add.Scalar)),
    ("aten_addcdiv", False, (aten.addcdiv.default,)),
    ("aten_addcmul", False, (aten.addcmul.default,)),
    ("aten_alias", False, (aten.alias.default,)),
    ("aten_angle", False, (aten.angle.default,)),
    ("aten_angle_complex", True, (aten.angle.default,)),
    ("aten_asin", False, (aten.asin.default,)),
    ("aten_asinh", False, (aten.asinh.default,)),
    ("aten_atan", False, (aten.atan.default,)),
    ("aten_atan2", False, (aten.atan2.default,)),
    ("aten_atanh", False, (aten.atanh.default,)),
    ("aten_bitwise_and", False, (aten.bitwise_and.Tensor, operator.and_)),
    ("aten_bitwise_and_scalar", False, (aten.bitwise_and.Scalar,)),
    ("aten_bitwise_and_scalar_tensor", False, (aten.bitwise_and.Scalar_Tensor,)),
    (
        "aten_bitwise_left_shift",
        False,
        (aten.bitwise_left_shift.Tensor, aten.__lshift__.Tensor, operator.lshift),
    ),
    (
        "aten_bitwise_left_shift_tensor_scalar",
        False,
        (aten.bitwise_left_shift.Tensor_Scalar, aten.__lshift__.Scalar),
    ),
    (
        "aten_bitwise_left_shift_scalar_tensor",
        False,
        (aten.bitwise_left_shift.Scalar_Tensor,),
    ),
    ("aten_bitwise_not", False, (aten.bitwise_not.default,)),
    ("aten_bitwise_or", False, (aten.bitwise_or.Tensor, operator.or_)),
    ("aten_bitwise_or_scalar", False, (aten.bitwise_or.Scalar,)),
    ("aten_bitwise_or_scalar_tensor", False, (aten.bitwise_or.Scalar_Tensor,)),
    (
        "aten_bitwise_right_shift",
        False,
        (aten.bitwise_right_shift.Tensor, aten.__rshift__.Tensor, operator.rshift),
    ),
    (
        "aten_bitwise_right_shift_tensor_scalar",
        False,
        (aten.bitwise_right_shift.Tensor_Scalar, aten.__rshift__.Scalar),
    ),
    (
        "aten_bitwise_right_shift_scalar_tensor",
        False,
        (aten.bitwise_right_shift.Scalar_Tensor,),
    ),
    ("aten_bitwise_xor", False, (aten.bitwise_xor.Tensor,)),
    ("aten_bitwise_xor_scalar", False, (aten.bitwise_xor.Scalar,)),
    ("aten_bitwise_xor_scalar_tensor", False, (aten.bitwise_xor.Scalar_Tensor,)),
    ("aten_ceil", False, (aten.ceil.default,)),
    ("python_math_ceil", False, (math.ceil,)),
    ("aten_clamp", False, (aten.clamp.default,)),
    ("aten_clamp_tensor", False, (aten.clamp.Tensor,)),
    ("aten_clamp_max", False, (aten.clamp_max.default,)),
    ("aten_clamp_max_tensor", False, (aten.clamp_max.Tensor,)),
    ("aten_clamp_min", False, (aten.clamp_min.default,)),
    ("aten_clamp_min_tensor", False, (aten.clamp_min.Tensor,)),
    ("aten_clone", False, (aten.clone.default,)),
    ("aten_clone_complex", True, (aten.clone.default,)),
    ("aten_conj", False, (aten.conj.default, aten._conj.default)),
    ("aten_conj_complex", True, (aten.conj.default, aten._conj.default)),
    ("aten__to_copy", False, (aten._to_copy.default,)),
    ("aten_cos", False, (aten.cos.default,)),
    ("aten_cosh", False, (aten.cosh.default,)),
    ("aten_deg2rad", False, (aten.deg2rad.default,)),
    (
        "aten_div",
        False,
        (
            aten.div.Tensor,
            aten.div.Scalar,
            aten.divide.Tensor,
            aten.divide.Scalar,
            aten.true_divide.Tensor,
            aten.true_divide.Scalar,
        ),
    ),
    ("operator_truediv", False, (operator.truediv,)),
    (
        "aten_div_complex",
        True,
        (
            aten.div.Tensor,
            aten.div.Scalar,
            aten.divide.Tensor,
            aten.divide.Scalar,
            aten.true_divide.Tensor,
            aten.true_divide.Scalar,
        ),
    ),
    ("aten_div_mode", False, (aten.div.Tensor_mode, aten.div.Scalar_mode)),
    ("aten_eq", False, (aten.eq.default, aten.eq.Tensor, aten.eq.Scalar, operator.eq)),
    ("aten_exp", False, (aten.exp.default,)),
    ("aten_exp2", False, (aten.exp2.default,)),
    ("aten_floor", False, (aten.floor.default,)),
    ("python_math_floor", False, (math.floor,)),
    ("aten_floor_divide", False, (aten.floor_divide.default,)),
    ("operator_floordiv", False, (operator.floordiv,)),
    ("aten_fmod", False, (aten.fmod.Tensor, aten.fmod.Scalar)),
    ("aten_frac", False, (aten.frac.default,)),
    ("aten_ge", False, (aten.ge.Tensor, aten.ge.Scalar, aten.greater_equal.Tensor)),
    ("operator_ge", False, (operator.ge,)),
    ("aten_gt", False, (aten.gt.Tensor, aten.gt.Scalar, aten.greater.Tensor)),
    ("operator_gt", False, (operator.gt,)),
    ("aten_isclose", False, (aten.isclose.default,)),
    ("aten_isfinite", False, (aten.isfinite.default,)),
    ("aten_isinf", False, (aten.isinf.default,)),
    ("aten_isnan", False, (aten.isnan.default,)),
    ("aten_isneginf", False, (aten.isneginf.default,)),
    ("aten_isposinf", False, (aten.isposinf.default,)),
    ("aten_le", False, (aten.le.Tensor, aten.le.Scalar, aten.less_equal.Tensor)),
    ("operator_le", False, (operator.le,)),
    ("aten_lerp", False, (aten.lerp.Tensor, aten.lerp.Scalar)),
    ("aten_log", False, (aten.log.default,)),
    ("aten_log10", False, (aten.log10.default,)),
    ("aten_log1p", False, (aten.log1p.default,)),
    ("aten_log2", False, (aten.log2.default,)),
    ("aten_logaddexp", False, (aten.logaddexp.default,)),
    ("aten_logaddexp2", False, (aten.logaddexp2.default,)),
    ("aten_logical_and", False, (aten.logical_and.default,)),
    ("aten_logical_not", False, (aten.logical_not.default,)),
    ("aten_logical_or", False, (aten.logical_or.default,)),
    ("aten_logical_xor", False, (aten.logical_xor.default,)),
    ("aten_logit", False, (aten.logit.default,)),
    ("aten_lt", False, (aten.lt.Tensor, aten.lt.Scalar, aten.less.Tensor)),
    ("operator_lt", False, (operator.lt,)),
    ("aten_maximum", False, (aten.maximum.default,)),
    ("aten_minimum", False, (aten.minimum.default,)),
    ("aten_mul", False, (aten.mul.default, aten.mul.Tensor, aten.multiply.Tensor)),
    ("operator_mul", False, (operator.mul,)),
    (
        "aten_mul_complex",
        True,
        (aten.mul.default, aten.mul.Tensor, aten.multiply.Tensor),
    ),
    ("aten_mul_scalar", False, (aten.mul.Scalar, aten.multiply.Scalar)),
    ("aten_ne", False, (aten.ne.default, aten.ne.Scalar, aten.ne.Tensor, operator.ne)),
    ("aten_neg", False, (aten.neg.default, operator.neg)),
    ("aten_neg_complex", True, (aten.neg.default, operator.neg)),
    ("aten_pow", False, (aten.pow.Tensor_Tensor, operator.pow)),
    ("aten_pow_tensor_scalar", False, (aten.pow.Tensor_Scalar,)),
    ("aten_pow_scalar", False, (aten.pow.Scalar,)),
    ("aten_rad2deg", False, (aten.rad2deg.default,)),
    ("aten_reciprocal", False, (aten.reciprocal.default,)),
    ("aten_reciprocal_complex", True, (aten.reciprocal.default,)),
    ("aten_remainder", False, (aten.remainder.Tensor, prims.remainder.default)),
    ("aten_remainder_scalar", False, (aten.remainder.Scalar,)),
    ("aten_remainder_scalar_tensor", False, (aten.remainder.Scalar_Tensor,)),
    ("operator_mod", False, (operator.mod,)),
    ("aten_round", False, (aten.round.default,)),
    ("aten_round_decimals", False, (aten.round.decimals,)),
    ("aten_rsqrt", False, (aten.rsqrt.default,)),
    ("aten_sign", False, (aten.sign.default,)),
    ("aten_sin", False, (aten.sin.default,)),
    ("aten_sinh", False, (aten.sinh.default,)),
    ("aten_sqrt", False, (aten.sqrt.default,)),
    ("aten_sub", False, (aten.sub.Tensor, aten.subtract.Tensor, operator.sub)),
    ("aten_sub_scalar", False, (aten.sub.Scalar, aten.subtract.Scalar)),
    (
        "aten_sub_complex",
        True,
        (aten.sub.Tensor, aten.sub.Scalar, aten.subtract.Tensor, aten.subtract.Scalar),
    ),
    ("aten_tan", False, (aten.tan.default,)),
    ("aten_tanh", False, (aten.tanh.default,)),
    ("aten_trunc", False, (aten.trunc.default,)),
    ("python_math_trunc", False, (math.trunc,)),
    ("aten_type_as", False, (aten.type_as.default,)),
)


def _evaluate(function, tensors, *, opset=18):
    graph = ir.Graph([], [], nodes=[], name="leaf", opset_imports={"": opset})
    builder = _ir_builder.GraphBuilder(graph, opset=opset)
    values = [
        ir.Value(
            name=f"input_{i}",
            shape=ir.Shape(tensor.shape),
            type=ir.TensorType(ir.DataType.from_numpy(tensor.numpy().dtype)),
        )
        for i, tensor in enumerate(tensors)
    ]
    graph.inputs.extend(values)
    with no_script(), builder.activate():
        result = function(*values)
    graph.outputs.append(result)
    proto = onnx.shape_inference.infer_shapes(
        ir.to_proto(ir.Model(graph, ir_version=10)), strict_mode=True
    )
    onnx.checker.check_model(proto, full_check=True)
    result = ReferenceEvaluator(proto).run(
        None, {value.name: tensor.numpy() for value, tensor in zip(values, tensors)}
    )[0]
    return torch.from_numpy(np.asarray(result))


class TestNativeLeafOps(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    def test_invalid_inputs_are_explicit(self, device):
        x = torch.tensor([1], dtype=torch.int64, device=device)
        y = torch.tensor([1], dtype=torch.int32, device=device)
        with self.assertRaisesRegex(ValueError, "matching tensor dtypes.*INT64.*INT32"):
            _evaluate(core.aten_bitwise_and, [x, y])
        with self.assertRaisesRegex(
            ValueError, "rounding_mode='nearest'.*trunc.*floor"
        ):
            _evaluate(lambda a, b: core.aten_div_mode(a, b, "nearest"), [x, x])
        with self.assertRaisesRegex(NotImplementedError, "integer or boolean"):
            _evaluate(core.aten_bitwise_not, [x.float()])
        with self.assertRaisesRegex(TypeError, "Use COMPLEX64 or COMPLEX128"):
            _evaluate(
                lambda value: core.aten_scalar_tensor_complex(
                    value, dtype=ir.DataType.FLOAT
                ),
                [x.float()],
            )

    @common_utils.parametrize("opset", [18, 20, 23])
    def test_source_aliases_resolve_to_native(self, device, opset):
        registry = _registration.ONNXRegistry.from_torchlib(opset_version=opset)
        seen = set()
        for name, complex_input, targets in _EXPECTED_CORE_TARGETS:
            for target in targets:
                key = (target, complex_input)
                self.assertNotIn(key, seen)
                seen.add(key)
                entries = [
                    entry
                    for entry in registry.get_decomps(target)
                    if entry.is_complex == complex_input
                ]
                self.assertEqual(len(entries), 1, (name, target, complex_input))
                translation = entries[0].onnx_function
                self.assertIsInstance(
                    translation, _custom_translation.NativeTranslation
                )
                self.assertIs(translation.function, getattr(core, name))
                self.assertFalse(entries[0].is_custom)
                self.assertEqual(entries[0].opset_introduced, 18)
        self.assertEqual(len(seen), 180)

    @common_utils.parametrize("shape", [(), (3,), (0, 3)])
    @common_utils.parametrize(
        "name",
        [
            "add_scalar",
            "sub_scalar",
            "mul_scalar",
            "remainder_scalar",
            "remainder_scalar_tensor",
            "bitwise_and_scalar",
            "bitwise_and_scalar_tensor",
            "bitwise_or_scalar",
            "bitwise_or_scalar_tensor",
            "bitwise_xor_scalar",
            "bitwise_xor_scalar_tensor",
            "bitwise_left_shift_tensor_scalar",
            "bitwise_left_shift_scalar_tensor",
            "bitwise_right_shift_tensor_scalar",
            "bitwise_right_shift_scalar_tensor",
        ],
    )
    def test_scalar_aliases(self, device, shape, name):
        x = torch.full(shape, 2, dtype=torch.int64, device=device)
        function = getattr(core, f"aten_{name}")
        if name.endswith("_scalar_tensor"):
            actual = _evaluate(lambda value: function(7, value), [x])
            if name.startswith("remainder"):
                expected = torch.remainder(7, x)
            else:
                expected = getattr(torch, name.removesuffix("_scalar_tensor"))(7, x)
        else:
            actual = _evaluate(lambda value: function(value, 3), [x])
            torch_name = name.removesuffix("_tensor_scalar").removesuffix("_scalar")
            expected = getattr(torch, torch_name)(x, 3)
        self.assertEqual(actual, expected)

    @common_utils.parametrize(
        "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
    )
    @common_utils.parametrize("mode", ["trunc", "floor", None])
    def test_division_boundaries(self, device, dtype, mode):
        x = torch.tensor([-7, -6, -1, 0, 1, 6, 7], dtype=dtype, device=device)
        y = torch.tensor([3, -3, 3, 3, -3, 3, -3], dtype=dtype, device=device)
        self.assertEqual(
            _evaluate(lambda a, b: core.aten_div_mode(a, b, mode), [x, y]),
            torch.div(x, y, rounding_mode=mode),
        )
        self.assertEqual(
            _evaluate(core.aten_floor_divide, [x, y]), torch.floor_divide(x, y)
        )
        self.assertEqual(_evaluate(core.aten_remainder, [x, y]), torch.remainder(x, y))
        self.assertEqual(_evaluate(core.aten_fmod, [x, y]), torch.fmod(x, y))

    @common_utils.parametrize("base", [2, 2.5])
    @common_utils.parametrize(
        "dtype", [torch.bool, torch.int64, torch.float32, torch.float64]
    )
    def test_scalar_pow(self, device, base, dtype):
        exponent = torch.tensor([0, 1, 1], dtype=dtype, device=device)
        self.assertEqual(
            _evaluate(lambda value: core.aten_pow_scalar(base, value), [exponent]),
            torch.pow(base, exponent),
        )
        if dtype != torch.bool:
            x = torch.tensor([1, 2, 3], dtype=dtype, device=device)
            self.assertEqual(
                _evaluate(lambda value: core.aten_pow_tensor_scalar(value, base), [x]),
                torch.pow(x, base),
            )

    def test_scalar_pow_preserves_double_precision(self, device):
        exponent = torch.tensor([1], dtype=torch.float64, device=device)
        base = 1.23456789012345
        self.assertEqual(
            _evaluate(lambda value: core.aten_pow_scalar(base, value), [exponent]),
            torch.pow(base, exponent),
            atol=0,
            rtol=0,
        )

    @common_utils.parametrize("shape", [(), (2,), (0, 2)])
    @common_utils.parametrize(
        "onnx_dtype",
        [
            ir.DataType.INT64,
            ir.DataType.DOUBLE,
            ir.DataType.COMPLEX64,
            ir.DataType.COMPLEX128,
        ],
    )
    def test_cast_and_copy(self, device, shape, onnx_dtype):
        x = torch.full(shape, 2.5, dtype=torch.float32, device=device)
        torch_dtype = {
            ir.DataType.INT64: torch.int64,
            ir.DataType.DOUBLE: torch.float64,
            ir.DataType.COMPLEX64: torch.complex64,
            ir.DataType.COMPLEX128: torch.complex128,
        }[onnx_dtype]
        expected = x.to(torch_dtype)
        if expected.is_complex():
            expected = torch.view_as_real(expected)
        for function in (common.cast_to, core.aten__to_copy):
            self.assertEqual(
                _evaluate(lambda value: function(value, dtype=onnx_dtype), [x]),
                expected,
            )
        self.assertEqual(_evaluate(core.aten__to_copy, [x]), x)

    def test_type_as_and_symbolic_dimensions(self, device):
        x = torch.tensor(3.0, device=device)
        y = torch.tensor(2, dtype=torch.int64, device=device)
        self.assertEqual(_evaluate(core.aten_type_as, [x, y]), x.type_as(y))
        self.assertEqual(
            _evaluate(lambda value: common.merge_dims([2, value, 4]), [y]),
            torch.tensor([2, 2, 4], dtype=torch.int64),
        )
        self.assertEqual(
            _evaluate(lambda: common.merge_dims([]), []),
            torch.empty(0, dtype=torch.int64),
        )
        with self.assertRaisesRegex(TypeError, "Cast dimensions to INT64"):
            _evaluate(lambda value: common.merge_dims([value]), [x])

    @common_utils.parametrize(
        "onnx_dtype",
        [
            ir.DataType.FLOAT,
            ir.DataType.DOUBLE,
            ir.DataType.COMPLEX64,
            ir.DataType.COMPLEX128,
        ],
    )
    def test_scalar_tensor(self, device, onnx_dtype):
        scalar = 1.23456789012345
        expected_dtype = {
            ir.DataType.FLOAT: torch.float32,
            ir.DataType.DOUBLE: torch.float64,
            ir.DataType.COMPLEX64: torch.complex64,
            ir.DataType.COMPLEX128: torch.complex128,
        }[onnx_dtype]
        expected = torch.scalar_tensor(scalar, dtype=expected_dtype, device=device)
        if expected.is_complex():
            expected = torch.view_as_real(expected)
        self.assertEqual(
            _evaluate(lambda: core.aten_scalar_tensor(scalar, dtype=onnx_dtype), []),
            expected,
        )
        if expected_dtype in (torch.complex64, torch.complex128):
            x = torch.tensor([1.25, -2.0], device=device)
            self.assertEqual(
                _evaluate(
                    lambda value: core.aten_scalar_tensor_complex(
                        value, dtype=onnx_dtype
                    ),
                    [x],
                ),
                x.to(
                    torch.float64
                    if expected_dtype == torch.complex128
                    else torch.float32
                ),
            )

    @common_utils.parametrize("eps", [None, 0.0, 0.2, 0.5, 0.6])
    def test_logit_boundaries(self, device, eps):
        x = torch.tensor(
            [-1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, float("nan")], device=device
        )
        self.assertEqual(
            _evaluate(lambda value: core.aten_logit(value, eps), [x]),
            torch.logit(x, eps=eps),
            equal_nan=True,
        )

    @common_utils.parametrize("equal_nan", [False, True])
    def test_isclose_nonfinite(self, device, equal_nan):
        x = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 1.0, 1.01, float("inf")],
            device=device,
        )
        y = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 1.0, 1.0, 0.0], device=device
        )
        self.assertEqual(
            _evaluate(
                lambda a, b: core.aten_isclose(a, b, equal_nan=equal_nan), [x, y]
            ),
            torch.isclose(x, y, equal_nan=equal_nan),
        )

    @common_utils.parametrize("name", ["add", "sub", "mul", "div"])
    @common_utils.parametrize("empty", [False, True])
    def test_complex_binary(self, device, name, empty):
        x = torch.tensor([[2 + 3j], [-4 + 2j]], dtype=torch.complex128, device=device)
        y = torch.tensor([1 - 2j, 3 + 4j], dtype=torch.complex128, device=device)
        if empty:
            x = x[:0]
        actual = _evaluate(
            getattr(core, f"aten_{name}_complex"),
            [torch.view_as_real(x), torch.view_as_real(y)],
        )
        self.assertEqual(actual, torch.view_as_real(getattr(torch, name)(x, y)))

    @common_utils.parametrize("name", ["add", "sub", "mul", "div"])
    @common_utils.parametrize("scalar", [2, 2.5, 2 + 3j])
    def test_complex_python_scalar(self, device, name, scalar):
        x = torch.tensor([2 + 3j, -4 + 2j], dtype=torch.complex128, device=device)
        function = getattr(core, f"aten_{name}_complex")
        self.assertEqual(
            _evaluate(lambda value: function(value, scalar), [torch.view_as_real(x)]),
            torch.view_as_real(getattr(torch, name)(x, scalar)),
        )

    @common_utils.parametrize("optimize", [False, True])
    @common_utils.parametrize("complex_input", [False, True])
    def test_export_never_calls_legacy_translation(
        self, device, optimize, complex_input
    ):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return ((x * y + x) / y - y).abs()

        x = torch.tensor([-3.0, 2.0], device=device)
        y = torch.tensor([2.0, 4.0], device=device)
        if complex_input:
            x = torch.complex(x, x.flip(0))
            y = torch.complex(y, y.flip(0))
        model = Model().eval()
        _registration.ONNXRegistry.from_torchlib()
        with (
            no_script(),
            mock.patch.object(
                _core,
                "_call_legacy_translation",
                side_effect=AssertionError("legacy translation called"),
            ),
        ):
            program = torch.onnx.export(
                model, (x, y), opset_version=20, optimize=optimize, verbose=False
            )
        proto = ir.to_proto(program.model)
        onnx.checker.check_model(proto, full_check=True)
        inputs = [
            torch.view_as_real(value) if complex_input else value for value in (x, y)
        ]
        actual = ReferenceEvaluator(proto).run(
            None,
            {
                value.name: tensor.numpy()
                for value, tensor in zip(program.model.graph.inputs, inputs)
            },
        )[0]
        self.assertEqual(torch.from_numpy(actual), model(x, y))


instantiate_device_type_tests(TestNativeLeafOps, globals(), only_for=["cpu"])

if __name__ == "__main__":
    common_utils.run_tests()
