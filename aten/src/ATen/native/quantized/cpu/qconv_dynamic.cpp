#include <algorithm>
#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/SmallVector.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
// #include <Aten/native/quantized/cpu/qconv.cpp> //do i need this or is it imported from library
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>
#include <c10/util/irange.h>

// #ifdef USE_FBGEMM

// template <int kSpatialDim>
// at::Tensor PackedConvWeight<kSpatialDim>::apply_dynamic(
//     const at::Tensor input,
//     bool reduce_range) {
  
//   /// is this needed?
//   //auto input_contig = input.contiguous();
//   const auto* input_ptr = input_contig.data_ptr<float>();

//   TORCH_CHECK(
//       fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");

//   float x_min, x_max;
//   fbgemm::FindMinMax(
//       /*m=*/input_ptr,
//       /*min=*/&x_min,
//       /*max=*/&x_max,
//       /*len=*/input.numel());

//   // Input tensor is quantized as 8-bit unsigned values
//   static constexpr int precision = 8;
//   static constexpr bool is_signed = false;

//   // Calculate scale and zero point for quantization of input tensor
//   auto q_params = quant_utils::ChooseQuantizationParams(
//       /*min=*/x_min,
//       /*max=*/x_max,
//       /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
//       /*qmax=*/
//       is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
//       /*preserve_sparsity=*/false,
//       /*force_scale_power_of_two=*/false,
//       /*reduce_range=*/reduce_range);

//   q_params.precision = precision;

//   return apply_impl<false>(input, output_scale, output_zero_point);}

// #endif // USE_FBGEMM

namespace at {
namespace native {
namespace {

class QConv1dDynamicInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      bool reduce_range) {
    at::Tensor output;
    // N, C, L -> N, C, 1, L
    input = input.unsqueeze(quant_utils::kConv1dSqueezeDim + 2); 
    output = QConvDynamicInt8<2>::run(std::move(input), packed_weight, reduce_range);
    // N, C, 1, L -> N, C, L
    return output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
    
  }
};

// class QConv1dDynamicInt8 final {
//  public:
//   static at::Tensor run(
//       at::Tensor input,
//       const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
//       bool reduce_range) {

//     // On empty input, no output data will be generated,
//     // so use arbitrary qparams.
//     float x_min = 0;
//     float x_max = 0;
//     // Otherwise...
//     if (input.numel() > 0) {
//       x_min = input.min().item<float>();
//       x_max = input.max().item<float>();
//     }

//     // Input tensor is quantized as 8-bit unsigned values
//     static constexpr int precision = 8;
//     static constexpr bool is_signed = false;

//     // Calculate scale and zero point for quantization of input tensor
//     auto q_params = quant_utils::ChooseQuantizationParams(
//         /*min=*/x_min,
//         /*max=*/x_max,
//         /*qmin=*/is_signed ? -(1 << (precision - 1)) : 0,
//         /*qmax=*/
//         is_signed ? ((1 << (precision - 1)) - 1) : (1 << precision) - 1,
//         /*preserve_sparsity=*/false,
//         /*force_scale_power_of_two=*/false,
//         /*reduce_range=*/reduce_range);

//     // Quantize input
//     at::Tensor q_input = at::quantize_per_tensor(
//         input, q_params.scale, q_params.zero_point, c10::kQUInt8);

//     auto out = quantized::conv1d(q_input, packed_weight, qparams.scale, qparams.zero_point) //how do i call this function????
    
//     return at::dequantize(out)
//   }
// };

template <int kSpatialDim>
class QConvDynamicInt8 final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      bool reduce_range) {
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
        /*reduce_range=*/reduce_range);

    // Quantize input
    at::Tensor q_input = at::quantize_per_tensor(
        input, q_params.scale, q_params.zero_point, c10::kQUInt8);

    at::Tensor out;
    if (kSpatialDim == 1) {
      auto out = quantized::conv1d(q_input, packed_weight, q_params.scale, q_params.zero_point);
    } else {
      // does this actually work?
      auto out = QConvInt8<kSpatialDim, false>::run(q_input, packed_weight, q_params.scale, q_params.zero_point);
      // quantized::conv2d.new(q_input, packed_weight, q_params.scale, q_params.zero_point)
      // quantized::conv3d.new(q_input, packed_weight, q_params.scale, q_params.zero_point)
    }

    return at::dequantize(out);
  
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_dynamic"), TORCH_FN(QConv1dDynamicInt8::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_dynamic"), TORCH_FN(QConvDynamicInt8<2>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_dynamic"), TORCH_FN(QConvDynamicInt8<3>::run));

  // // transpose
  // m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_dynamic"),  TORCH_FN(QConvTranspose1dDynamicInt8::run));
  // m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_dynamic"),  TORCH_FN(QConvTranspose2dDynamicInt8::run));
  // m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_dynamic"),  TORCH_FN(QConvTranspose3dDynamicInt8::run));
}

/* is this needed?
TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_dynamic"), TORCH_FN(QLinearDynamicInt8<false>::run));
}*/

} // namespace
} // namespace native
} // namespace at
