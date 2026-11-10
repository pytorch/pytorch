# Owner(s): ["module: onnx"]
"""Final native families and the native-only built-in registry cutover."""

from __future__ import annotations

import builtins
import contextlib
from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator
from test_native_factory_sequence_ops import _evaluate
from test_native_torchlib import no_script

import torch
from torch.onnx._internal._lazy_import import onnxscript_apis
from torch.onnx._internal.exporter import (
    _core,
    _custom_translation,
    _ir_builder,
    _registration,
    _schemas,
)
from torch.onnx._internal.exporter._torchlib import _torchlib_registry
from torch.onnx._internal.exporter._torchlib.ops import (
    core,
    fft,
    linalg,
    neural,
    nn,
    prims,
    quantized_decomposed,
    special,
    vision,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


# Recursive torch_lib source snapshot at 97749eb0 (including nested decorators).
_SOURCE_KEYS = {
    False: [
        "_operator::__lshift__",
        "_operator::__rshift__",
        "_operator::abs",
        "_operator::add",
        "_operator::and_",
        "_operator::eq",
        "_operator::floordiv",
        "_operator::ge",
        "_operator::getitem",
        "_operator::gt",
        "_operator::le",
        "_operator::lt",
        "_operator::mod",
        "_operator::mul",
        "_operator::ne",
        "_operator::neg",
        "_operator::or_",
        "_operator::pow",
        "_operator::sub",
        "_operator::truediv",
        "aten::__lshift__.Scalar",
        "aten::__lshift__.Tensor",
        "aten::__rshift__.Scalar",
        "aten::__rshift__.Tensor",
        "aten::_conj",
        "aten::_embedding_bag",
        "aten::_embedding_bag_forward_only",
        "aten::_fft_r2c",
        "aten::_grouped_mm",
        "aten::_linalg_det",
        "aten::_local_scalar_dense",
        "aten::_log_softmax",
        "aten::_native_batch_norm_legit",
        "aten::_native_batch_norm_legit.no_stats",
        "aten::_native_batch_norm_legit_functional",
        "aten::_native_batch_norm_legit_no_training",
        "aten::_prelu_kernel",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_flash_attention_for_cpu",
        "aten::_softmax",
        "aten::_to_copy",
        "aten::_unique",
        "aten::_unique2",
        "aten::_unsafe_index.Tensor",
        "aten::_unsafe_index_put",
        "aten::_unsafe_view",
        "aten::_upsample_bicubic2d_aa",
        "aten::_upsample_bilinear2d_aa",
        "aten::abs",
        "aten::acos",
        "aten::acosh",
        "aten::add.Scalar",
        "aten::add.Tensor",
        "aten::addbmm",
        "aten::addcdiv",
        "aten::addcmul",
        "aten::addmm",
        "aten::addmv",
        "aten::addr",
        "aten::alias",
        "aten::all",
        "aten::all.dim",
        "aten::all.dims",
        "aten::allclose",
        "aten::amax",
        "aten::amin",
        "aten::angle",
        "aten::any",
        "aten::any.dim",
        "aten::any.dims",
        "aten::arange",
        "aten::arange.start",
        "aten::arange.start_step",
        "aten::argmax",
        "aten::argmin",
        "aten::as_strided",
        "aten::asin",
        "aten::asinh",
        "aten::atan",
        "aten::atan2",
        "aten::atanh",
        "aten::atleast_1d",
        "aten::atleast_1d.Sequence",
        "aten::atleast_2d",
        "aten::atleast_2d.Sequence",
        "aten::atleast_3d",
        "aten::atleast_3d.Sequence",
        "aten::avg_pool1d",
        "aten::avg_pool2d",
        "aten::avg_pool3d",
        "aten::baddbmm",
        "aten::bernoulli",
        "aten::bernoulli.p",
        "aten::bilinear",
        "aten::bincount",
        "aten::bitwise_and.Scalar",
        "aten::bitwise_and.Scalar_Tensor",
        "aten::bitwise_and.Tensor",
        "aten::bitwise_left_shift.Scalar_Tensor",
        "aten::bitwise_left_shift.Tensor",
        "aten::bitwise_left_shift.Tensor_Scalar",
        "aten::bitwise_not",
        "aten::bitwise_or.Scalar",
        "aten::bitwise_or.Scalar_Tensor",
        "aten::bitwise_or.Tensor",
        "aten::bitwise_right_shift.Scalar_Tensor",
        "aten::bitwise_right_shift.Tensor",
        "aten::bitwise_right_shift.Tensor_Scalar",
        "aten::bitwise_xor.Scalar",
        "aten::bitwise_xor.Scalar_Tensor",
        "aten::bitwise_xor.Tensor",
        "aten::blackman_window",
        "aten::bmm",
        "aten::broadcast_to",
        "aten::cat",
        "aten::ceil",
        "aten::celu",
        "aten::chunk",
        "aten::clamp",
        "aten::clamp.Tensor",
        "aten::clamp_max",
        "aten::clamp_max.Tensor",
        "aten::clamp_min",
        "aten::clamp_min.Tensor",
        "aten::clone",
        "aten::col2im",
        "aten::complex",
        "aten::concat",
        "aten::concatenate",
        "aten::conj",
        "aten::constant_pad_nd",
        "aten::contiguous",
        "aten::conv1d",
        "aten::conv2d",
        "aten::conv3d",
        "aten::convolution",
        "aten::copy",
        "aten::cos",
        "aten::cosh",
        "aten::cross",
        "aten::cross_entropy_loss",
        "aten::cumsum",
        "aten::deg2rad",
        "aten::det",
        "aten::detach",
        "aten::diagonal",
        "aten::diagonal_copy",
        "aten::div.Scalar",
        "aten::div.Scalar_mode",
        "aten::div.Tensor",
        "aten::div.Tensor_mode",
        "aten::divide.Scalar",
        "aten::divide.Tensor",
        "aten::dot",
        "aten::dropout",
        "aten::einsum",
        "aten::elu",
        "aten::embedding",
        "aten::embedding_bag",
        "aten::embedding_bag.padding_idx",
        "aten::embedding_renorm",
        "aten::empty.memory_format",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::eq",
        "aten::eq.Scalar",
        "aten::eq.Tensor",
        "aten::equal",
        "aten::erf",
        "aten::erfc",
        "aten::exp",
        "aten::exp2",
        "aten::expand",
        "aten::expand_as",
        "aten::expm1",
        "aten::fake_quantize_per_channel_affine",
        "aten::fake_quantize_per_tensor_affine",
        "aten::fake_quantize_per_tensor_affine.tensor_qparams",
        "aten::fill.Scalar",
        "aten::fill.Tensor",
        "aten::flatten.using_ints",
        "aten::flip",
        "aten::floor",
        "aten::floor_divide",
        "aten::fmod.Scalar",
        "aten::fmod.Tensor",
        "aten::frac",
        "aten::full",
        "aten::full_like",
        "aten::gather",
        "aten::ge.Scalar",
        "aten::ge.Tensor",
        "aten::gelu",
        "aten::getitem",
        "aten::glu",
        "aten::greater.Tensor",
        "aten::greater_equal.Tensor",
        "aten::grid_sampler",
        "aten::grid_sampler_2d",
        "aten::group_norm",
        "aten::gru.input",
        "aten::gt.Scalar",
        "aten::gt.Tensor",
        "aten::hamming_window",
        "aten::hann_window",
        "aten::hardsigmoid",
        "aten::hardswish",
        "aten::hardtanh",
        "aten::hardtanh_backward",
        "aten::heaviside",
        "aten::histc",
        "aten::im2col",
        "aten::index.Tensor",
        "aten::index_put",
        "aten::index_select",
        "aten::instance_norm",
        "aten::is_nonzero",
        "aten::isclose",
        "aten::isfinite",
        "aten::isinf",
        "aten::isnan",
        "aten::isneginf",
        "aten::isposinf",
        "aten::layer_norm",
        "aten::le.Scalar",
        "aten::le.Tensor",
        "aten::leaky_relu",
        "aten::lerp.Scalar",
        "aten::lerp.Tensor",
        "aten::less.Tensor",
        "aten::less_equal.Tensor",
        "aten::lift_fresh_copy",
        "aten::linalg_cross",
        "aten::linalg_det",
        "aten::linalg_vector_norm",
        "aten::linear",
        "aten::linspace",
        "aten::log",
        "aten::log10",
        "aten::log1p",
        "aten::log2",
        "aten::log_sigmoid",
        "aten::log_softmax.int",
        "aten::logaddexp",
        "aten::logaddexp2",
        "aten::logcumsumexp",
        "aten::logdet",
        "aten::logical_and",
        "aten::logical_not",
        "aten::logical_or",
        "aten::logical_xor",
        "aten::logit",
        "aten::logsumexp",
        "aten::lstm.input",
        "aten::lt.Scalar",
        "aten::lt.Tensor",
        "aten::mH",
        "aten::mT",
        "aten::masked_fill.Scalar",
        "aten::masked_fill.Tensor",
        "aten::masked_scatter",
        "aten::matmul",
        "aten::max",
        "aten::max.dim",
        "aten::max_pool1d",
        "aten::max_pool1d_with_indices",
        "aten::max_pool2d",
        "aten::max_pool2d_with_indices",
        "aten::max_pool3d",
        "aten::max_pool3d_with_indices",
        "aten::maximum",
        "aten::mean",
        "aten::mean.dim",
        "aten::min",
        "aten::min.dim",
        "aten::minimum",
        "aten::mish",
        "aten::mm",
        "aten::mse_loss",
        "aten::mul",
        "aten::mul.Scalar",
        "aten::mul.Tensor",
        "aten::multinomial",
        "aten::multiply.Scalar",
        "aten::multiply.Tensor",
        "aten::mv",
        "aten::narrow",
        "aten::native_batch_norm",
        "aten::native_dropout",
        "aten::native_group_norm",
        "aten::native_layer_norm",
        "aten::ne",
        "aten::ne.Scalar",
        "aten::ne.Tensor",
        "aten::neg",
        "aten::new_empty",
        "aten::new_empty_strided",
        "aten::new_full",
        "aten::new_ones",
        "aten::new_zeros",
        "aten::nll_loss",
        "aten::nll_loss_forward",
        "aten::nonzero",
        "aten::normal.Tensor_Tensor",
        "aten::normal.Tensor_float",
        "aten::normal.float_Tensor",
        "aten::normal.float_float",
        "aten::normal_functional",
        "aten::ones",
        "aten::ones_like",
        "aten::pad",
        "aten::permute",
        "aten::pixel_shuffle",
        "aten::pixel_unshuffle",
        "aten::polar",
        "aten::pow.Scalar",
        "aten::pow.Tensor_Scalar",
        "aten::pow.Tensor_Tensor",
        "aten::prelu",
        "aten::prod",
        "aten::prod.dim_int",
        "aten::rad2deg",
        "aten::rand",
        "aten::rand_like",
        "aten::randint",
        "aten::randint.low",
        "aten::randint_like",
        "aten::randint_like.low_dtype",
        "aten::randn",
        "aten::randn_like",
        "aten::reciprocal",
        "aten::reflection_pad1d",
        "aten::reflection_pad2d",
        "aten::reflection_pad3d",
        "aten::relu",
        "aten::relu6",
        "aten::remainder.Scalar",
        "aten::remainder.Scalar_Tensor",
        "aten::remainder.Tensor",
        "aten::repeat",
        "aten::repeat_interleave.Tensor",
        "aten::repeat_interleave.self_int",
        "aten::replication_pad1d",
        "aten::replication_pad2d",
        "aten::replication_pad3d",
        "aten::reshape",
        "aten::resolve_conj",
        "aten::resolve_neg",
        "aten::roll",
        "aten::round",
        "aten::round.decimals",
        "aten::rsqrt",
        "aten::scalar_tensor",
        "aten::scaled_dot_product_attention",
        "aten::scatter.src",
        "aten::scatter.value",
        "aten::scatter_add",
        "aten::scatter_reduce.two",
        "aten::select.int",
        "aten::select_scatter",
        "aten::selu",
        "aten::sigmoid",
        "aten::sign",
        "aten::signbit",
        "aten::silu",
        "aten::sin",
        "aten::sinc",
        "aten::sinh",
        "aten::slice.Tensor",
        "aten::slice_scatter",
        "aten::softmax.int",
        "aten::softplus",
        "aten::sort",
        "aten::special_erf",
        "aten::special_erfc",
        "aten::special_erfcx",
        "aten::special_expm1",
        "aten::special_log_softmax",
        "aten::special_sinc",
        "aten::special_softmax",
        "aten::split",
        "aten::split.Tensor",
        "aten::split_with_sizes",
        "aten::sqrt",
        "aten::squeeze",
        "aten::squeeze.dim",
        "aten::stack",
        "aten::stft",
        "aten::sub.Scalar",
        "aten::sub.Tensor",
        "aten::subtract.Scalar",
        "aten::subtract.Tensor",
        "aten::sum",
        "aten::sum.dim_IntList",
        "aten::sym_size.int",
        "aten::sym_storage_offset",
        "aten::t",
        "aten::tan",
        "aten::tanh",
        "aten::tensor.bool",
        "aten::tensor.float",
        "aten::tensor.int",
        "aten::tile",
        "aten::topk",
        "aten::transpose.int",
        "aten::tril",
        "aten::triu",
        "aten::true_divide.Scalar",
        "aten::true_divide.Tensor",
        "aten::trunc",
        "aten::type_as",
        "aten::unbind.int",
        "aten::unflatten.int",
        "aten::unfold",
        "aten::unique_consecutive",
        "aten::unique_dim",
        "aten::unsafe_split.Tensor",
        "aten::unsqueeze",
        "aten::upsample_bicubic2d",
        "aten::upsample_bicubic2d.vec",
        "aten::upsample_bilinear2d",
        "aten::upsample_bilinear2d.vec",
        "aten::upsample_linear1d",
        "aten::upsample_linear1d.vec",
        "aten::upsample_nearest1d",
        "aten::upsample_nearest1d.vec",
        "aten::upsample_nearest2d",
        "aten::upsample_nearest2d.vec",
        "aten::upsample_nearest3d",
        "aten::upsample_nearest3d.vec",
        "aten::upsample_trilinear3d",
        "aten::upsample_trilinear3d.vec",
        "aten::view",
        "aten::view_as",
        "aten::view_as_complex",
        "aten::view_as_complex_copy",
        "aten::view_copy",
        "aten::where.Scalar",
        "aten::where.ScalarOther",
        "aten::where.ScalarSelf",
        "aten::where.self",
        "aten::xlogy.Scalar_Other",
        "aten::xlogy.Scalar_Self",
        "aten::xlogy.Tensor",
        "aten::zeros",
        "aten::zeros_like",
        "math::ceil",
        "math::floor",
        "math::trunc",
        "prims::abs",
        "prims::acos",
        "prims::acosh",
        "prims::add",
        "prims::asin",
        "prims::asinh",
        "prims::atan",
        "prims::atanh",
        "prims::broadcast_in_dim",
        "prims::ceil",
        "prims::convert_element_type",
        "prims::cos",
        "prims::cosh",
        "prims::device_put",
        "prims::div",
        "prims::eq",
        "prims::erf",
        "prims::exp",
        "prims::floor",
        "prims::ge",
        "prims::gt",
        "prims::le",
        "prims::log",
        "prims::lt",
        "prims::mul",
        "prims::ne",
        "prims::neg",
        "prims::pow",
        "prims::remainder",
        "prims::reshape",
        "prims::resize",
        "prims::round",
        "prims::sin",
        "prims::sinh",
        "prims::sqrt",
        "prims::squeeze",
        "prims::sub",
        "prims::sum",
        "prims::tan",
        "prims::tanh",
        "prims::transpose",
        "prims::var",
        "prims::where",
        "quantized_decomposed::dequantize_per_channel",
        "quantized_decomposed::dequantize_per_channel.tensor",
        "quantized_decomposed::dequantize_per_channel.tensor2",
        "quantized_decomposed::dequantize_per_tensor",
        "quantized_decomposed::dequantize_per_tensor.tensor",
        "quantized_decomposed::dequantize_per_tensor.tensor2",
        "quantized_decomposed::quantize_per_channel",
        "quantized_decomposed::quantize_per_channel.tensor",
        "quantized_decomposed::quantize_per_channel.tensor2",
        "quantized_decomposed::quantize_per_tensor",
        "quantized_decomposed::quantize_per_tensor.tensor",
        "quantized_decomposed::quantize_per_tensor.tensor2",
        "torchvision::deform_conv2d",
        "torchvision::nms",
        "torchvision::roi_align",
        "torchvision::roi_pool",
    ],
    True: [
        "_operator::neg",
        "aten::_conj",
        "aten::_fft_c2c",
        "aten::_fft_c2r",
        "aten::_unsafe_view",
        "aten::abs",
        "aten::add.Scalar",
        "aten::add.Tensor",
        "aten::angle",
        "aten::cat",
        "aten::clone",
        "aten::concat",
        "aten::concatenate",
        "aten::conj",
        "aten::conv1d",
        "aten::conv2d",
        "aten::conv3d",
        "aten::div.Scalar",
        "aten::div.Tensor",
        "aten::divide.Scalar",
        "aten::divide.Tensor",
        "aten::flip",
        "aten::mH",
        "aten::mT",
        "aten::matmul",
        "aten::mean",
        "aten::mean.dim",
        "aten::mul",
        "aten::mul.Tensor",
        "aten::multiply.Tensor",
        "aten::neg",
        "aten::permute",
        "aten::reciprocal",
        "aten::roll",
        "aten::scalar_tensor",
        "aten::slice.Tensor",
        "aten::squeeze.dim",
        "aten::stack",
        "aten::sub.Scalar",
        "aten::sub.Tensor",
        "aten::subtract.Scalar",
        "aten::subtract.Tensor",
        "aten::sum",
        "aten::sum.dim_IntList",
        "aten::t",
        "aten::transpose.int",
        "aten::true_divide.Scalar",
        "aten::true_divide.Tensor",
        "aten::unsqueeze",
        "aten::view",
        "aten::view_as_real",
        "aten::view_as_real_copy",
    ],
}


@contextlib.contextmanager
def native_only():
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if (
            "torch_lib" in name
            or name == "torchvision"
            or name.startswith("torchvision.")
        ):
            raise AssertionError(
                f"Built-in translation imported optional/legacy {name}"
            )
        return original_import(name, *args, **kwargs)

    with (
        no_script(),
        mock.patch.object(builtins, "__import__", side_effect=checked_import),
        mock.patch.object(
            onnxscript_apis, "get_torchlib_ops", side_effect=AssertionError("fallback")
        ),
        mock.patch.object(
            _schemas,
            "op_signature_from_function",
            side_effect=AssertionError("signature"),
        ),
        mock.patch.object(
            _core, "_call_legacy_translation", side_effect=AssertionError("legacy")
        ),
    ):
        yield


class TestNativeFinalOps(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    @common_utils.parametrize("opset", [18, 20, 23])
    @common_utils.parametrize("normalization", [0, 1, 2])
    @common_utils.parametrize("forward", [False, True])
    def test_fft_complex(self, device, opset, normalization, forward):
        x = torch.randn(3, 4, dtype=torch.complex64, device=device)
        expected = torch.view_as_real(
            torch.ops.aten._fft_c2c(x, [0, 1], normalization, forward)
        )
        actual = _evaluate(
            lambda a: fft.aten__fft_c2c(a, [0, 1], normalization, forward),
            [torch.view_as_real(x)],
            expected,
            opset=opset,
        )
        self.assertEqual(actual, expected, atol=2e-6, rtol=2e-5)

    @common_utils.parametrize("normalization", [0, 1, 2])
    @common_utils.parametrize("onesided", [False, True])
    @common_utils.parametrize("dimensions", [[0], [1], [0, 1]])
    def test_fft_real(self, device, normalization, onesided, dimensions):
        x = torch.randn(3, 4, device=device)
        expected = torch.view_as_real(
            torch.ops.aten._fft_r2c(x, dimensions, normalization, onesided)
        )
        actual = _evaluate(
            lambda a: fft.aten__fft_r2c(a, dimensions, normalization, onesided),
            [x],
            expected,
        )
        self.assertEqual(actual, expected, atol=2e-6, rtol=2e-5)

    @common_utils.parametrize("normalization", [0, 1, 2])
    @common_utils.parametrize("length", [4, 5])
    def test_fft_inverse_real(self, device, normalization, length):
        x = torch.fft.rfft(torch.randn(2, length, device=device))
        expected = torch.ops.aten._fft_c2r(x, [1], normalization, length)
        actual = _evaluate(
            lambda a: fft.aten__fft_c2r(a, [1], normalization, length),
            [torch.view_as_real(x)],
            expected,
        )
        self.assertEqual(actual, expected, atol=2e-6, rtol=2e-5)

    @common_utils.parametrize(
        "name",
        [
            "abs",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "ceil",
            "cos",
            "cosh",
            "erf",
            "exp",
            "floor",
            "log",
            "neg",
            "round",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
        ],
    )
    def test_prims_unary(self, device, name):
        x = torch.tensor([0.2, 0.4, 0.7], device=device)
        if name == "acosh":
            x += 1
        expected = getattr(torch, name)(x)
        actual = _evaluate(getattr(prims, "prims_" + name), [x], expected)
        self.assertEqual(actual, expected)

    @common_utils.parametrize(
        "name", ["add", "div", "eq", "ge", "gt", "le", "lt", "mul", "ne", "pow", "sub"]
    )
    def test_prims_binary(self, device, name):
        x, y = torch.rand(2, 3, device=device), torch.rand(2, 3, device=device) + 1
        expected = getattr(torch, name)(x, y)
        actual = _evaluate(getattr(prims, "prims_" + name), [x, y], expected)
        self.assertEqual(actual, expected)

    def test_prims_metadata(self, device):
        x = torch.randn(2, 1, 3, device=device)
        cases = [
            (lambda a: prims.prims_reshape(a, [3, 2]), x.reshape(3, 2)),
            (lambda a: prims.prims_squeeze(a, [1]), x.squeeze(1)),
            (lambda a: prims.prims_transpose(a, [2, 0, 1]), x.permute(2, 0, 1)),
            (lambda a: prims.prims_resize(a, [2, 4, 3]), x.expand(2, 4, 3)),
            (lambda a: prims.prims_device_put(a, "cpu"), x),
            (
                lambda a: prims.prims_convert_element_type(a, ir.DataType.DOUBLE),
                x.double(),
            ),
            (
                lambda a: prims.prims_sum(a, [0, 2], ir.DataType.DOUBLE),
                x.sum((0, 2)).double(),
            ),
        ]
        for function, expected in cases:
            with self.subTest(function=function):
                self.assertEqual(_evaluate(function, [x], expected), expected)
        self.assertEqual(
            _evaluate(
                lambda a: prims.prims_where(
                    _ir_builder.current_builder().call("Greater", a, 0)[0],
                    a,
                    _ir_builder.current_builder().call("Neg", a)[0],
                ),
                [x],
                x.abs(),
            ),
            x.abs(),
        )

    @common_utils.parametrize("dims", [None, [], [0], [0, 1]])
    @common_utils.parametrize("correction", [0, 1])
    def test_prims_var(self, device, dims, correction):
        x = torch.randn(3, 4, device=device)
        expected = torch.var(x, dim=dims, correction=correction)
        actual = _evaluate(
            lambda a: prims.prims_var(a, dims, correction), [x], expected
        )
        self.assertEqual(actual, expected)

    def test_symbolic_broadcast_shape(self, device):
        x = torch.randn(3, device=device)
        expected = x.expand(2, 3)
        actual = _evaluate(
            lambda a: prims.prims_broadcast_in_dim(
                a, [2, _ir_builder.current_builder().call("Size", a)[0]], [1]
            ),
            [x],
            expected,
        )
        self.assertEqual(actual, expected)

    @common_utils.parametrize("name", ["erf", "erfc", "erfcx", "expm1", "sinc"])
    def test_special(self, device, name):
        x = torch.tensor([-1.0, -0.1, 0.0, 0.2, 1.0], device=device)
        expected = getattr(torch.special, name)(x)
        actual = _evaluate(getattr(special, "aten_special_" + name), [x], expected)
        self.assertEqual(actual, expected, atol=1e-6, rtol=1e-5)

    def test_special_scalar_variants(self, device):
        x = torch.tensor([0.2, 1.0, 2.0], device=device)
        for function, expected in (
            (lambda a: special.aten_special_xlogy(2.0, a), torch.xlogy(2.0, x)),
            (lambda a: special.aten_special_xlogy(a, 2.0), torch.xlogy(x, 2.0)),
            (lambda a: special.aten_special_log_softmax(a, 0), x.log_softmax(0)),
        ):
            self.assertEqual(_evaluate(function, [x], expected), expected)

    @common_utils.parametrize("ord", [0, 1, 2, 3, -2, float("inf"), -float("inf")])
    def test_linalg(self, device, ord):
        x = torch.randn(2, 3, device=device)
        expected = torch.linalg.vector_norm(x, ord, dim=[1], keepdim=True)
        self.assertEqual(
            _evaluate(
                lambda a: linalg.aten_linalg_vector_norm(a, ord, [1], True),
                [x],
                expected,
            ),
            expected,
        )
        matrix = torch.randn(2, 3, 3, device=device)
        determinant = torch.linalg.det(matrix)
        self.assertEqual(
            _evaluate(linalg.aten_linalg_det, [matrix], determinant), determinant
        )

    @common_utils.parametrize("correction", [0, 1])
    @common_utils.parametrize("dim", [None, [1]])
    @common_utils.parametrize("keepdim", [False, True])
    def test_unregistered_source_variance_helpers(
        self, device, correction, dim, keepdim
    ):
        x = torch.randn(3, 4, device=device)
        for name, reference in (
            ("var_correction", torch.var),
            ("std_correction", torch.std),
            ("var_mean_correction", torch.var_mean),
            ("std_mean_correction", torch.std_mean),
        ):
            expected = reference(x, dim=dim, correction=correction, keepdim=keepdim)
            actual = _evaluate(
                lambda a: getattr(core, "aten_" + name)(a, dim, correction, keepdim),
                [x],
                expected,
            )
            self.assertEqual(
                actual, list(expected) if isinstance(expected, tuple) else expected
            )

    @common_utils.parametrize("dtype", [torch.int8, torch.uint8])
    @common_utils.parametrize("out_dtype", [-1, ir.DataType.DOUBLE])
    def test_dequantization(self, device, dtype, out_dtype):
        lo, hi = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        x = torch.tensor([[lo, 1, hi], [4, 7, 9]], dtype=dtype, device=device)
        scales = torch.tensor([0.25, 0.5], dtype=torch.float64, device=device)
        zero_points = torch.tensor([1, 2], device=device)
        onnx_dtype = ir.DataType.INT8 if dtype == torch.int8 else ir.DataType.UINT8
        expected = (x.float() - zero_points[:, None]) * scales.float()[:, None]
        if out_dtype == ir.DataType.DOUBLE:
            expected = expected.double()
        actual = _evaluate(
            lambda a,
            s,
            z: quantized_decomposed.quantized_decomposed_dequantize_per_channel(
                a, s, z, 0, 0, 127, onnx_dtype, out_dtype
            ),
            [x, scales, zero_points],
            expected,
            runtime=True,
        )
        self.assertEqual(actual, expected)
        expected = (x.float() - 2) * 0.25
        if out_dtype == ir.DataType.DOUBLE:
            expected = expected.double()
        self.assertEqual(
            _evaluate(
                lambda a: quantized_decomposed.quantized_decomposed_dequantize_per_tensor(
                    a, 0.25, 2, 0, 127, onnx_dtype, out_dtype
                ),
                [x],
                expected,
                runtime=True,
            ),
            expected,
        )

    @common_utils.parametrize("compute_log_sumexp", [False, True])
    def test_attention_auxiliary_outputs(self, device, compute_log_sumexp):
        q = torch.randn(2, 3, 5, 4, device=device)
        k, v = torch.randn_like(q), torch.randn_like(q)
        result = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # Source auxiliary outputs are zero placeholders, not actual statistics/RNG state.
        efficient = (
            result,
            torch.zeros(2, 3, 32 if compute_log_sumexp else 0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        actual = _evaluate(
            lambda q, k, v: neural.aten__scaled_dot_product_efficient_attention(
                q, k, v, None, compute_log_sumexp
            ),
            [q, k, v],
            efficient,
        )
        self.assertEqual(actual, list(efficient), atol=1e-6, rtol=1e-5)
        flash = (
            result,
            torch.zeros(2, 3, 5),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        actual = _evaluate(
            neural.aten__scaled_dot_product_flash_attention, [q, k, v], flash
        )
        self.assertEqual(actual, list(flash), atol=1e-6, rtol=1e-5)

    def test_vision_without_optional_package(self, device):
        boxes = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [3.0, 3.0, 4.0, 4.0]],
            device=device,
        )
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        self.assertEqual(
            _evaluate(
                lambda b, s: vision.torchvision_nms(b, s, 0.5),
                [boxes, scores],
                torch.tensor([0, 2]),
                runtime=True,
            ),
            torch.tensor([0, 2]),
        )
        x = torch.arange(16.0, device=device).reshape(1, 1, 4, 4)
        rois = torch.tensor([[0.0, 0.0, 0.0, 3.0, 3.0]], device=device)
        self.assertEqual(
            _evaluate(
                lambda a, r: vision.torchvision_roi_align(a, r, 1.0, 1, 1, 1, False),
                [x, rois],
                torch.tensor([[[[7.5]]]]),
                runtime=True,
            ),
            torch.tensor([[[[7.5]]]]),
        )
        self.assertEqual(
            _evaluate(
                lambda a, r: vision.torchvision_roi_pool(a, r, [1, 1]),
                [x, rois],
                torch.tensor([[[[15.0]]]]),
                runtime=True,
            ),
            torch.tensor([[[[15.0]]]]),
        )
        graph = ir.Graph([], [], nodes=[], name="deform_conv", opset_imports={"": 19})
        builder = _ir_builder.GraphBuilder(graph, opset=19)
        values = [
            ir.Value(name=n, type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(s))
            for n, s in (
                ("x", [1, 1, 4, 4]),
                ("w", [1, 1, 1, 1]),
                ("offset", [1, 2, 4, 4]),
            )
        ]
        graph.inputs.extend(values)
        with builder.activate():
            output = vision.torchvision_deform_conv2d(
                *values, None, None, 1, 1, 0, 0, 1, 1, 1, 1, False
            )
        output.shape = ir.Shape([1, 1, 4, 4])
        graph.outputs.append(output)
        onnx.checker.check_model(
            ir.to_proto(ir.Model(graph, ir_version=10)), full_check=True
        )


@common_utils.instantiate_parametrized_tests
class TestNativeCutover(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    def test_optional_namespace_loaded_after_native_import(self):
        with (
            mock.patch.object(
                _torchlib_registry, "_registry", list(_torchlib_registry._registry)
            ),
            torch.library._scoped_library("torchvision", "FRAGMENT") as library,
        ):
            library.define("_native_onnx_probe(Tensor input) -> Tensor")
            with native_only():
                _torchlib_registry.onnx_impl("torchvision::_native_onnx_probe")(
                    lambda x: x
                )
                registry = _registration.ONNXRegistry.from_torchlib(
                    custom_translations=()
                )
                target = torch.ops.torchvision._native_onnx_probe.default
                entry = registry.get_decomps(target)[0]
                self.assertIs(entry.fx_target, target)

    def test_all_supported_source_keys(self):
        self.assertEqual(sum(map(len, _SOURCE_KEYS.values())), 561)
        with native_only():
            registry = _registration.ONNXRegistry.from_torchlib(
                19, custom_translations=()
            )
            for complex_, names in _SOURCE_KEYS.items():
                for name in names:
                    with self.subTest(target=name, complex=complex_):
                        target = _registration._get_overload(name)
                        entries = registry.get_decomps(
                            name if target is None else target
                        )
                        selected = [
                            entry for entry in entries if entry.is_complex == complex_
                        ]
                        self.assertEqual(len(selected), 1)
                        self.assertIsInstance(
                            selected[0].onnx_function,
                            _custom_translation.NativeTranslation,
                        )

    def test_registry_is_entirely_native_and_opset_aware(self):
        with native_only():
            for opset in (18, 19, 20, 21, 23):
                registry = _registration.ONNXRegistry.from_torchlib(opset)
                for entries in registry.functions.values():
                    self.assertTrue(entries)
                    for entry in entries:
                        self.assertIsInstance(
                            entry.onnx_function, _custom_translation.NativeTranslation
                        )
                        self.assertIsNone(entry.signature)
                        self.assertLessEqual(entry.opset_introduced, opset)
                target = torch.ops.aten.scaled_dot_product_attention.default
                expected = (
                    nn.aten_scaled_dot_product_attention_23
                    if opset >= 23
                    else neural.aten_scaled_dot_product_attention
                )
                self.assertIs(
                    registry.get_decomps(target)[0].onnx_function.function, expected
                )
                self.assertEqual(
                    bool(registry.get_decomps("torchvision::deform_conv2d")),
                    opset >= 19,
                )
            self.assertIs(
                _registration.ONNXRegistry.from_torchlib()
                .get_decomps("aten::getitem")[0]
                .onnx_function.function,
                _registration.ONNXRegistry.from_torchlib()
                .get_decomps(__import__("operator").getitem)[0]
                .onnx_function.function,
            )

    @common_utils.parametrize("optimize", [False, True])
    def test_native_export_blocks_legacy_construction(self, optimize):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.special.erf(x + y).reshape(-1).sin()

        model = Model().eval()
        x, y = torch.randn(2, 3), torch.randn(2, 3)
        with (
            native_only(),
            mock.patch.object(
                onnxscript_apis, "optimize", wraps=onnxscript_apis.optimize
            ) as optimizer,
        ):
            program = torch.onnx.export(
                model,
                (x, y),
                dynamo=True,
                optimize=optimize,
                dynamic_shapes=(
                    {0: torch.export.Dim("batch")},
                    {0: torch.export.Dim("batch")},
                ),
                verbose=False,
            )
            self.assertEqual(optimizer.call_count, int(optimize))
        onnx.checker.check_model(program.model_proto, full_check=True)
        actual = ReferenceEvaluator(program.model_proto).run(
            None,
            {
                value.name: tensor.numpy()
                for value, tensor in zip(program.model.graph.inputs, (x, y))
            },
        )[0]
        self.assertEqual(torch.from_numpy(np.asarray(actual).copy()), model(x, y))


instantiate_device_type_tests(TestNativeFinalOps, globals(), only_for="cpu")


if __name__ == "__main__":
    common_utils.run_tests()
