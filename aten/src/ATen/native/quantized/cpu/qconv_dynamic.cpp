#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <ATen/Parallel.h>
#include <ATen/SmallVector.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/dequantize.h>                           // for dequantize
#include <ATen/ops/quantize_per_tensor.h>
#endif

#ifdef USE_FBGEMM

template <int kSpatialDim>
at::Tensor PackedConvWeight<kSpatialDim>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range) {
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

  float x_min, x_max;
  fbgemm::FindMinMax(
      /*m=*/input.data_ptr<float>(),
      /*min=*/&x_min,
      /*max=*/&x_max,
      /*len=*/input.numel());

  // Input tensor is quantized as 8-bit unsigned values
  static constexpr int precision = 8;
  static constexpr bool is_signed = false;

  // Calculate scale and zero point for quantization of input tensor
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
      /*qmax=*/
      is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  // Quantize input
  at::Tensor q_input = at::quantize_per_tensor(
      input, q_params.scale, q_params.zero_point, c10::kQUInt8);

  at::Tensor out =
      apply_impl<false>(q_input, q_params.scale, q_params.zero_point);

  return at::dequantize(out); // TODO: optimized kernel that outputs fp32 so
                              // this step isn't necessary
}

template at::Tensor PackedConvWeight<2>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range);

template at::Tensor PackedConvWeight<3>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range);

#endif // USE_FBGEMM

#ifdef USE_PYTORCH_QNNPACK

template <int kSpatialDim>
at::Tensor PackedConvWeightsQnnp<kSpatialDim>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range) {
  if (reduce_range) {
    TORCH_WARN("Currently, qnnpack incorrectly ignores reduce_range when it is set to true; this may change in a future release.");
  }

  // On empty input, no output data will be generated,
  // so use arbitrary qparams.
  float x_min = 0;
  float x_max = 0;
  // Otherwise...
  if (input.numel() > 0) {
    x_min = input.min().item<float>();
    x_max = input.max().item<float>();
  }

  // Input tensor is quantized as 8-bit unsigned values
  static constexpr int precision = 8;
  static constexpr bool is_signed = false;

  // Calculate scale and zero point for quantization of input tensor
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
      /*qmax=*/
      is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/false); // note: this is set to false rather than
                               // reduce_range for qnnpack

  // Quantize input
  at::Tensor q_input = at::quantize_per_tensor(
      input, q_params.scale, q_params.zero_point, c10::kQUInt8);

  at::Tensor out =
      apply_impl<false>(q_input, q_params.scale, q_params.zero_point);

  return at::dequantize(out); // TODO: optimized kernel that outputs fp32 so
                              // this step isn't necessary
}

template at::Tensor PackedConvWeightsQnnp<2>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range);

template at::Tensor PackedConvWeightsQnnp<3>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range);

#endif // USE_PYTORCH_QNNPACK

#if AT_MKLDNN_ENABLED()

template <int kSpatialDim>
at::Tensor PackedConvWeightsOnednn<kSpatialDim>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range) {

  // Find min/max of input
  float x_max = 0, x_min = 0;
  if (input.numel() > 0) {
    x_min = input.min().item<float>();
    x_max = input.max().item<float>();
  }

  // Input tensor is quantized as 8-bit unsigned values
  static constexpr int precision = 8;
  static constexpr bool is_signed = false;

  // Calculate scale and zero point for quantization of input tensor
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
      /*qmax=*/
      is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);

  // Quantize input
  at::Tensor q_input = at::quantize_per_tensor(
      input, q_params.scale, q_params.zero_point, c10::kQUInt8);

  at::Tensor out =
      apply_impl<false>(q_input, /*accum*/std::nullopt, q_params.scale, q_params.zero_point);

  // TODO: Modify ideep to allow fp32 input & output
  // to avoid explicit `quantize - dequantize`
  return at::dequantize(out);
}

template at::Tensor PackedConvWeightsOnednn<2>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range);

template at::Tensor PackedConvWeightsOnednn<3>::apply_dynamic(
    const at::Tensor& input,
    bool reduce_range);

#endif // AT_MKLDNN_ENABLED()

namespace at::native {
namespace {

// note: this works for both Conv and ConvT due to transpose()
template <int kSpatialDim>
class QConvDynamicInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>&
          packed_weight,
      bool reduce_range) {
    return packed_weight->apply_dynamic(input, reduce_range);
  }
};

// note: this works for both Conv and ConvT due to transpose()
class QConv1dDynamicInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      bool reduce_range) {
    at::Tensor output;
    // N, C, L -> N, C, 1, L
    input = input.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    output = packed_weight->apply_dynamic(input, reduce_range);
    // N, C, 1, L -> N, C, L
    return output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv1d_dynamic"),
      TORCH_FN(QConv1dDynamicInt8::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_dynamic"),
      TORCH_FN(QConvDynamicInt8<2>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv3d_dynamic"),
      TORCH_FN(QConvDynamicInt8<3>::run));

  // transpose
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_dynamic"),
      TORCH_FN(QConv1dDynamicInt8::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_dynamic"),
      TORCH_FN(QConvDynamicInt8<2>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_dynamic"),
      TORCH_FN(QConvDynamicInt8<3>::run));
}

} // namespace
} // namespace at::native
