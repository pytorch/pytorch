#include <algorithm>
#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/SmallVector.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/qconv.cpp>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#include <torch/custom_class.h>

#include <string>

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

  q_params.precision = precision;

  // Quantize input
  at::Tensor act = at::quantize_per_tensor(
      input, q_params.scale, q_params.zero_point, c10::kQUInt8);

  const std::string func_name =
      transpose() ? "quantized::conv_transpose" : "quantized::conv";
  TORCH_CHECK(
      fbgemm::fbgemmSupportedCPU(), "Your CPU does not support FBGEMM.");
  ConvDimChecks<kSpatialDim>(
      act.ndimension(),
      stride().size(),
      padding().size(),
      output_padding().size(),
      dilation().size(),
      func_name,
      transpose());

  const int N = act.size(0);
  const int C = act.size(1);
  const int D = kSpatialDim == 2 ? 1 : act.size(2);
  const int H = act.size(kSpatialDim);
  const int W = act.size(kSpatialDim + 1);

  const at::Tensor act_ndhwc = kSpatialDim == 2
      ? act.contiguous(c10::MemoryFormat::ChannelsLast)
      : at::native::fbgemm_utils::ConvertToChannelsLast3dTensor(act);
  const uint8_t* act_data =
      reinterpret_cast<uint8_t*>(act_ndhwc.data_ptr<c10::quint8>());
  auto* pack_w = w.get();

  const int M = pack_w->outputChannels();
  const int kernel_d = kSpatialDim == 2 ? 1 : kernel[0];
  const int kernel_h = kernel[kSpatialDim - 2];
  const int kernel_w = kernel[kSpatialDim - 1];
  const int pad_d = kSpatialDim == 2 ? 0 : padding_[0];
  const int pad_h = padding_[kSpatialDim - 2];
  const int pad_w = padding_[kSpatialDim - 1];
  const int stride_d = kSpatialDim == 2 ? 1 : stride_[0];
  const int stride_h = stride_[kSpatialDim - 2];
  const int stride_w = stride_[kSpatialDim - 1];
  const int dilation_d = kSpatialDim == 2 ? 1 : dilation_[0];
  const int dilation_h = dilation_[kSpatialDim - 2];
  const int dilation_w = dilation_[kSpatialDim - 1];
  const int output_padding_d = kSpatialDim == 2 ? 0 : output_padding_[0];
  const int output_padding_h = output_padding_[kSpatialDim - 2];
  const int output_padding_w = output_padding_[kSpatialDim - 1];

  const float* bias_ptr = nullptr;
  at::Tensor bias_vec;
  if (bias.has_value()) {
    bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == N, //
        "bias should have N elements: " + std::to_string(N));
    // TODO: contiguous is called for further jit optimizations.
    auto bias_contig = bias_vec.contiguous();
    bias_ptr = bias_contig.data_ptr<float>();
  }

  if (kSpatialDim == 2) {
    TORCH_CHECK(
        C == pack_w->inputChannels(),
        "[QConv2D] Given groups=",
        groups_,
        ", weight of size ",
        M,
        ", ",
        kernel_h,
        ", ",
        kernel_w,
        ", ",
        pack_w->inputChannels(),
        ", expected input (NCHW) ",
        N,
        ", ",
        C,
        ", ",
        H,
        ", ",
        W,
        " to have ",
        pack_w->inputChannels(),
        " channels, but got ",
        C,
        " channels instead");
  } else {
    TORCH_CHECK(
        C == pack_w->inputChannels(),
        "[QConv3D] Given groups=",
        groups_,
        ", weight of size ",
        M,
        ", ",
        kernel_d,
        ", ",
        kernel_h,
        ", ",
        kernel_w,
        ", ",
        pack_w->inputChannels(),
        ", expected input (NCDHW) ",
        N,
        ", ",
        C,
        ", ",
        D,
        ", ",
        H,
        ", ",
        W,
        " to have ",
        pack_w->inputChannels(),
        " channels, but got ",
        C,
        " channels instead");
  }

  fbgemm::conv_param_t<kSpatialDim> conv_p =
      at::native::fbgemm_utils::MakeFbgemmConvParam<kSpatialDim>(
          N, // Batch size
          C, // Number of input channels
          M, // Number of output channels
          kSpatialDim == 2 ? std::vector<int>{H, W} : std::vector<int>{D, H, W},
          groups_,
          kSpatialDim == 2 ? std::vector<int>{kernel_h, kernel_w}
                           : std::vector<int>{kernel_d, kernel_h, kernel_w},
          kSpatialDim == 2 ? std::vector<int>{stride_h, stride_w}
                           : std::vector<int>{stride_d, stride_h, stride_w},
          kSpatialDim == 2 ? std::vector<int>{pad_h, pad_w}
                           : std::vector<int>{pad_d, pad_h, pad_w},
          kSpatialDim == 2
              ? std::vector<int>{dilation_h, dilation_w}
              : std::vector<int>{dilation_d, dilation_h, dilation_w},
          kSpatialDim == 2
              ? std::vector<int>{output_padding_h, output_padding_w}
              : std::vector<
                    int>{output_padding_d, output_padding_h, output_padding_w},
          transpose());

  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const float act_scale = act.q_scale();
  const int32_t act_zero_point = act.q_zero_point();

  at::Tensor bias;
  const float* bias_data = GetBiasData(&bias);

  TORCH_CHECK(
      w_scale.size() == w_zp.size(),
      "Weight scales and zero points vectors should have the same size.");
  std::vector<float> output_multiplier_float;
  std::vector<float> act_times_w_scale;

  //   GetQuantizationParams(
  //       act_scale, output_scale, &output_multiplier_float,
  //       &act_times_w_scale);

  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  if (transpose()) {
    output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W}
                         : std::vector<int64_t>{D, H, W},
        kernel,
        stride(),
        padding(),
        output_padding(),
        dilation());
  } else {
    output_shape = MakeConvOutputShape<kSpatialDim>(N, M, conv_p.OUT_DIM);
  }
  if (N > 0) {
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        "[QConv",
        kSpatialDim,
        "D] each dimension of output tensor should be greater than 0");
  }

  auto output = at::empty(out_sizes, input.options().dtype(at::kFloat));

  //   at::Tensor output = kSpatialDim == 2
  //       ? at::_empty_affine_quantized(
  //             output_shape,
  //             device(c10::kCPU)
  //                 .dtype(c10::kQUInt8)
  //                 .memory_format(c10::MemoryFormat::ChannelsLast),
  //             output_scale, ////////////
  //             output_zero_point,
  //             c10::nullopt)
  //       :
  //       at::native::fbgemm_utils::MakeEmptyAffineQuantizedChannelsLast3dTensor(
  //             output_shape[0],
  //             output_shape[1],
  //             output_shape[2],
  //             output_shape[3],
  //             output_shape[4],
  //             device(c10::kCPU).dtype(c10::kQUInt8),
  //             output_scale, ////////
  //             output_zero_point);

  at::Tensor buffer =
      at::empty(output.sizes(), output.options().dtype(c10::kInt));

  const int num_tasks = at::get_num_threads();
  at::parallel_for(0, num_tasks, 1, [&](int64_t begin, int64_t end) {
    fbgemm::DoNothing<> kNoOpObj{};
    for (const auto task_id : c10::irange(begin, end)) {
      fbgemm::ReQuantizeForFloat <false>
          outputProcObj(
              /*nextop=*/kNoOpObj,
              /*Aq_scale=*/q_params.scale,
              /*Bq_scale=*/w_scale.data(),
              /*Aq_zero_point=*/q_params.zero_point,
              /*Bq_zero_point=*/w_zp.data(),
              /*row_offsets=*/nullptr, // IS THIS RIGHT?
              /*col_offsets=*/col_offsets.data(),
              /*bias=*/bias_ptr,
              /*nCol=*/N); // WHAT GOES HERE?
      fbgemm::fbgemmConv<decltype(outputProcObj), kSpatialDim, int32_t>(
          conv_p,
          act_data,
          *pack_w,
          reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
          buffer.data_ptr<int32_t>(),
          outputProcObj,
          task_id /* thread_id*/,
          num_tasks /* num_threads */);
    }
  });

  return output;
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
    bool /*reduce_range*/) {
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

namespace at {
namespace native {
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
} // namespace native
} // namespace at
