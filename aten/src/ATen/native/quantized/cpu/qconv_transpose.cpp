#include <ATen/ATen.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/c10_utils.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <torch/library.h>

#include <vector>

namespace at {
namespace native {
namespace {

constexpr int64_t kReasonableMaxDim = 1000000;

inline int64_t compute_shape(int64_t input,
                             int64_t kernel,
                             int64_t stride,
                             int64_t input_padding,
                             int64_t output_padding,
                             int64_t dilation) {
  int64_t out = (input - 1) * stride - 2 * input_padding
                + dilation * (kernel - 1) + output_padding + 1;
  return out;
}

template <int64_t kSpatialDim>
SmallVector<int64_t, kSpatialDim + 2> MakeDeconvOutputShape(
    int64_t N, int64_t M,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& input_padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation) {
  SmallVector<int64_t, kSpatialDim + 2> output_shape;
  output_shape.resize(kSpatialDim + 2);
  output_shape[0] = N;
  output_shape[1] = M;
  for (int64_t idx = 0; idx < kSpatialDim; ++idx) {
    output_shape[idx + 2] = compute_shape(input_shape[idx],
                                      kernel[idx],
                                      stride[idx],
                                      input_padding[idx],
                                      output_padding[idx],
                                      dilation[idx]);
    TORCH_CHECK(output_shape[idx + 2] > 0,
                "Output dimension is zero for ", idx, " axis;"
                " kernel: ", kernel[idx],
                ", stride: ", stride[idx],
                ", input padding: ", input_padding[idx],
                ", output padding: ", output_padding[idx],
                ", dilation: ", dilation[idx])
    TORCH_CHECK(output_shape[idx + 2] < kReasonableMaxDim,
                "Output dimension is beyound reasonable maximum for ", idx,
                " axis;"
                " kernel: ", kernel[idx],
                ", stride: ", stride[idx],
                ", input padding: ", input_padding[idx],
                ", output padding: ", output_padding[idx],
                ", dilation: ", dilation[idx]);
  }
  return output_shape;
}

class QDeConv2dInt8 final {
 public:
  static Tensor run(Tensor act,
                    Tensor packed_weight,
                    torch::List<int64_t> stride,
                    torch::List<int64_t> input_padding,
                    torch::List<int64_t> output_padding,
                    torch::List<int64_t> dilation,
                    int64_t groups,
                    double output_scale,
                    int64_t output_zero_point) {
    auto& ctx = at::globalContext();
    TORCH_CHECK(stride[0] > 0 && stride[1] > 0);
    TORCH_CHECK(input_padding[0] >= 0 && input_padding[1] >= 0);
    TORCH_CHECK(output_padding[0] >= 0 && output_padding[1] >= 0);
    TORCH_CHECK(dilation[0] > 0 && dilation[1] > 0);
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return QnnpackDeconv(act,
                           packed_weight,
                           stride,
                           input_padding,
                           output_padding,
                           dilation,
                           groups,
                           output_scale,
                           output_zero_point);
    }
#else // Fallback (no engine)
    return FallBackDeconv(act,
                          packed_weight,
                          stride,
                          input_padding,
                          output_padding,
                          dilation,
                          groups,
                          output_scale,
                          output_zero_point);
#endif // USE_PYTORCH_QNNPACK

    TORCH_CHECK(false, "Didn't find engine for operation quantized::conv ",
                toString(ctx.qEngine()));
  }

 private:
#ifdef USE_PYTORCH_QNNPACK
  static at::Tensor QnnpackDeconv(Tensor act,
                       Tensor packed_weight,
                       torch::List<int64_t> stride,
                       torch::List<int64_t> input_padding,
                       torch::List<int64_t> output_padding,
                       torch::List<int64_t> dilation,
                       int64_t groups,
                       double output_scale,
                       int64_t output_zero_point) {
    PackedConvWeightsQnnp& pack_data =
        cpp_custom_type_hack::cast<PackedConvWeightsQnnp>(packed_weight);

    const auto* pack_w = pack_data.w.get();

    const int N = act.size(0);
    const int H = act.size(2);
    const int W = act.size(3);
    const int in_ch = act.size(1);
    const int M = pack_data.bias.size(0);  // output channels

    const std::vector<int64_t>& kernel = pack_data.kernel;
    const int64_t kernel_h = kernel[0];
    const int64_t kernel_w = kernel[1];
    const auto kernel_zp = pack_data.w_zp + 128;
    const auto& kernel_scale = pack_data.w_scale;

    const int64_t stride_h = stride[0];
    const int64_t stride_w = stride[1];
    const int64_t input_pad_t = input_padding[0];
    const int64_t input_pad_l = input_padding[1];
    const int64_t output_height_adjustment = output_padding[0];
    const int64_t output_width_adjustment = output_padding[1];
    const int64_t dilation_h = dilation[0];
    const int64_t dilation_w = dilation[1];

    qnnpack::conv_param_t deconv_p(
        {kernel_w, kernel_h},
        {stride_w, stride_h},
        {dilation_w, dilation_h},
        {input_pad_t, input_pad_l, input_pad_t, input_pad_l},
        {output_width_adjustment, output_height_adjustment},
        groups,
        in_ch,
        M,
        kernel_zp,
        kernel_scale,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        /*transpose=*/true);
    const Tensor& act_nhwc = act.contiguous(MemoryFormat::ChannelsLast);
    const auto input_scale = act_nhwc.q_scale();

    // Re-quantizing the bias based on input scale and weight scale.
    if (!pack_data.input_scale.has_value() ||
        pack_data.input_scale.value() != input_scale) {
      // Get the original weight and adjust it to uint8 from int8
      auto weight_contig =
          pack_data.orig_weight.contiguous(MemoryFormat::ChannelsLast);
      auto bias_fp32 = pack_data.bias;
      int8_t* w_data =
          reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());
      Tensor qnnp_weight = at::_empty_affine_quantized(
          weight_contig.sizes(),
          at::device(kCPU)
             .dtype(kQUInt8)
             .memory_format(MemoryFormat::ChannelsLast),
          kernel_scale,
          kernel_zp,
          c10::nullopt);
      auto* qnnp_w_data = qnnp_weight.data_ptr<c10::quint8>();
      auto wt_numel = weight_contig.numel();
      for (int i = 0; i < wt_numel; ++i) {
        qnnp_w_data[i] = static_cast<c10::quint8>(w_data[i] + 128);
      }
      // Original bias was float, so we requantize it here.
      auto bias = at::quantize_per_tensor(
          bias_fp32, kernel_scale * input_scale, 0, kQInt32);
      // Update the input scale to not pack again.
      pack_data.input_scale = input_scale;
      pack_data.w.reset();
      pack_data.w = std::make_unique<qnnpack::PrePackConvWeights>(
          deconv_p,
          reinterpret_cast<uint8_t*>(qnnp_w_data),
          reinterpret_cast<int32_t*>(bias.data_ptr<c10::qint32>()));
      pack_w = pack_data.w.get();
    }
    TORCH_INTERNAL_ASSERT(pack_w != nullptr, "Packed Weights are NULL");
    const auto output_shape = MakeDeconvOutputShape<2>(
        N, M, {H, W}, kernel, stride.vec(), input_padding.vec(),
        {output_height_adjustment, output_width_adjustment}, dilation.vec());
    TORCH_CHECK(
        std::all_of(
            output_shape.begin(),
            output_shape.end(),
            [](int64_t i) { return i > 0; }),
        "quantized::conv2d (qnnpack): each dimension of output tensor should "
        "be greater than 0.")
    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
        output_shape,
        at::device(kCPU)
           .dtype(kQUInt8)
           .memory_format(MemoryFormat::ChannelsLast),
        output_scale,
        output_zero_point,
        c10::nullopt);

    const pytorch_qnnp_status run_status = qnnpack::qnnpackDeConv(
        deconv_p,
        pack_w->getPackedWeights(),
        N,
        H,
        W,
        act_nhwc.q_scale(),
        act_nhwc.q_zero_point(),
        reinterpret_cast<uint8_t*>(act_nhwc.data_ptr<c10::quint8>()),
        output.q_scale(),
        output.q_zero_point(),
        reinterpret_cast<uint8_t*>(output.data_ptr<c10::quint8>()),
        caffe2::mobile_pthreadpool());

    TORCH_INTERNAL_ASSERT(
        run_status == pytorch_qnnp_status_success,
        "failed to run quantized::conv2d (qnnpack) operator");

    return output;
  }
#else // Fallback (no qengine)
  static at::Tensor FallBackDeconv(Tensor act,
                        Tensor packed_weight,
                        torch::List<int64_t> stride,
                        torch::List<int64_t> input_padding,
                        torch::List<int64_t> output_padding,
                        torch::List<int64_t> dilation,
                        int64_t groups,
                        double output_scale,
                        int64_t output_zero_point) {
    TORCH_WARN(
      "Using FallBack Deconvolution; check if the qengine is supported");
    const auto kUnpackFunctionName = "quantized::conv_transpose2d_unpack";
    std::vector<c10::IValue> unpacked_weight_list
        = callOp(kUnpackFunctionName, "", packed_weight);
    const Tensor weight = unpacked_weight_list[0].toTensor().dequantize();
    const Tensor bias = unpacked_weight_list[1].toTensor();
    const Tensor f_act = act.dequantize();

    const Tensor f_out = at::conv_transpose2d(f_act,
                                              weight,
                                              bias,
                                              stride.vec(),
                                              input_padding.vec(),
                                              output_padding.vec(),
                                              groups,
                                              dilation.vec());
    if (act.qscheme() == kPerTensorAffine) {
      return at::quantize_per_tensor(f_out, output_scale, output_zero_point,
                                     at::kQInt8);
    } else {
      TORCH_CHECK(false, "Only per tensor quantization is supported.");
    }
  }
#endif // USE_PYTORCH_QNNPACK
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl("conv_transpose2d", QDeConv2dInt8::run);
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl("conv_transpose2d", QDeConv2dInt8::run);
}

}  // namespace
}  // namespace native
}  // namespace at
