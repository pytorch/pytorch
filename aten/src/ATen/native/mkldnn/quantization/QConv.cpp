#include <algorithm>
#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/SmallVector.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/mkldnn/quantization/Utils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#include <c10/util/irange.h>

namespace {
// To have a sanity check for maximum matrix size.
constexpr int64_t kReasonableMaxDim = 1000000;

template <int kSpatialDim = 2>
bool ConvDimChecks(
    int64_t act_dims,
    int64_t stride_dims,
    int64_t padding_dims,
    int64_t output_padding_dims,
    int64_t dilation_dims,
    std::string func_name,
    bool transpose = false) {
  TORCH_CHECK(
      act_dims == kSpatialDim + 2,
      func_name,
      kSpatialDim,
      "d(): Expected activation tensor to have ",
      kSpatialDim + 2,
      " dimensions, got ",
      act_dims);
  TORCH_CHECK(
      stride_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected stride tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      stride_dims);
  TORCH_CHECK(
      padding_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected padding tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      padding_dims);
  TORCH_CHECK(
      !transpose || (output_padding_dims == kSpatialDim),
      func_name,
      kSpatialDim,
      "d(): Expected output padding tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      output_padding_dims);
  TORCH_CHECK(
      dilation_dims == kSpatialDim,
      func_name,
      kSpatialDim,
      "d(): Expected dilation tensor to have ",
      kSpatialDim,
      " dimensions, got ",
      dilation_dims);
  return true;
}

inline int64_t compute_deconv_shape(int64_t input,
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
at::SmallVector<int64_t, kSpatialDim + 2> MakeDeConvOutputShape(
    int64_t N, int64_t M,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel,
    const torch::List<int64_t>& stride,
    const torch::List<int64_t>& input_padding,
    const torch::List<int64_t>& output_padding,
    const torch::List<int64_t>& dilation) {
  at::SmallVector<int64_t, kSpatialDim + 2> output_shape;
  output_shape.resize(kSpatialDim + 2);
  output_shape[0] = N;  // Batch size
  output_shape[1] = M;  // Output channels
  for (int64_t idx = 0; idx < kSpatialDim; ++idx) {
    output_shape[idx + 2] = compute_deconv_shape(input_shape[idx],
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

} // namespace

#if AT_MKLDNN_ENABLED()
template <int kSpatialDim>
at::Tensor PackedConvWeightsMkldnn<kSpatialDim>::apply(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightsMkldnn<kSpatialDim>::apply_relu(
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, output_scale, output_zero_point);
}

template <int kSpatialDim>
template <bool kReluFused>
at::Tensor PackedConvWeightsMkldnn<kSpatialDim>::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  std::string func_name = "quantized::conv";
  if (transpose()) {
    func_name += "_transpose";
  }
  func_name += std::to_string(kSpatialDim) + "d";
  if (kReluFused) {
    func_name += "_relu";
  }
  ConvDimChecks<kSpatialDim>(
      act.ndimension(), stride().size(), padding().size(),
      output_padding().size(), dilation().size(), func_name, transpose());
  TORCH_CHECK(act.scalar_type() == c10::ScalarType::QUInt8,
      func_name, " (MKLDNN): data type of input should be QUint8.");

  // src
  auto act_contig = act.contiguous(kSpatialDim == 2 ? c10::MemoryFormat::ChannelsLast : c10::MemoryFormat::ChannelsLast3d);
  auto src_dims = act_contig.sizes().vec();
  auto src_data_type = dnnl::memory::data_type::u8;
  auto src_desc = ideep::tensor::desc(src_dims, src_data_type,
      kSpatialDim == 2 ? ideep::format_tag::nhwc : ideep::format_tag::ndhwc);
  ideep::tensor src;
  src.init(src_desc, act_contig.data_ptr());
  // weights & bias
  ideep::tensor& weights = *(weight_.get());
  bool with_bias = bias_.has_value();
  const auto& kernel_size = weights.get_dims();
  // dst
  const std::vector<int64_t>& input_size = src.get_dims();
  std::vector<int64_t> output_sizes;
  if (transpose()) {
    // Prepacked weight format: [o, i, ...]
    const bool with_groups = groups() > 1;
    const int N = act.size(0); // batch size
    const int C = act.size(1); // input channels
    const int M = weights.get_dim(0); // output channels
    const int D = kSpatialDim == 2 ? 1 : act.size(2); // input depth
    const int H = act.size(kSpatialDim); // input height
    const int W = act.size(kSpatialDim + 1); // input width
    const int KH = weights.get_dim(kSpatialDim); // kernel height
    const int KW = weights.get_dim(kSpatialDim + 1); // kernel width
    const int KD = kSpatialDim == 2 ? 1 : weights.get_dim(2); // kernel depth
    TORCH_CHECK(C == groups() * weights.get_dim(1), // weight: [o, i, ...]
                func_name, " (MKLDNN): input channel number should be ",
                groups() * weights.get_dim(1), ", but got ", C);
    auto output_shape = MakeDeConvOutputShape<kSpatialDim>(
        N,
        M,
        kSpatialDim == 2 ? std::vector<int64_t>{H, W} : std::vector<int64_t>{D, H, W},
        kSpatialDim == 2 ? std::vector<int64_t>{KH, KW} : std::vector<int64_t>{KD, KH, KW},
        stride(),
        padding(),
        output_padding(),
        dilation());
    output_sizes = c10::IntArrayRef(output_shape).vec();
  } else {
    output_sizes = at::native::conv_output_size(input_size, kernel_size, padding().vec(), stride().vec(), dilation().vec());
  }
  ideep::dims dst_dims = ideep::dims({output_sizes.cbegin(), output_sizes.cend()});
  at::Tensor output = at::_empty_affine_quantized(
      dst_dims,
      device(c10::kCPU)
          .dtype(c10::kQUInt8)
          .memory_format(kSpatialDim == 2 ?
              c10::MemoryFormat::ChannelsLast :
              c10::MemoryFormat::ChannelsLast3d),
      output_scale,
      output_zero_point,
      c10::nullopt);
  ideep::tensor dst({dst_dims, ideep::tensor::data_type::u8, {output.strides().cbegin(), output.strides().cend()}},
                    output.data_ptr());
  // Parameters
  const ideep::dims& strides = stride().vec();
  const ideep::dims& dilates = dilation().vec();
  const ideep::dims& padding_l = padding().vec();
  const ideep::dims& padding_r = padding().vec();
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/act.q_scale()); // Scales of MKLDNN and PyTorch are reciprocal
  const ideep::scale_t& weights_scales = weights.get_scale();
  const ideep::scale_t& dst_scales = ideep::scale_t(weights_scales.size(), 1.0/output_scale); // Scales of MKLDNN and PyTorch are reciprocal
  const ideep::zero_point_t src_zero_points = ideep::zero_point_t(1, act.q_zero_point());
  const ideep::zero_point_t dst_zero_points = ideep::zero_point_t(1, output_zero_point);
  ideep::attr_t op_attr = kReluFused ? ideep::attr_t::fuse_relu() : ideep::attr_t();
  op_attr.set_zero_points(DNNL_ARG_SRC, ideep::utils::tensor_zp_mask(1), {DNNL_RUNTIME_S32_VAL}); // runtime src zero point
  if (with_bias) {
    // Bias might be modified outside (e.g. by quantization bias correction).
    // If so, update the prepacked bias as well.
    if (bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
      bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
    }
    const auto& b = bias_.value();
    if (transpose()) {
      ideep::convolution_transpose_forward::compute_v2(
          src, weights, b, dst_dims, dst,
          strides, padding_l, padding_r, dilates,
          groups(), src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
          op_attr, dnnl::algorithm::deconvolution_direct, dnnl::prop_kind::forward_inference,
          ideep::u8s8, ideep::engine::cpu_engine());
    } else {
      ideep::convolution_forward::compute_v2(
          src, weights, b, dst_dims, dst,
          strides, dilates, padding_l, padding_r, groups(),
          src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
          op_attr, dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
          ideep::u8s8, ideep::engine::cpu_engine());
    }
  } else {
    if (transpose()) {
      ideep::convolution_transpose_forward::compute_v2(
          src, weights, dst_dims, dst,
          strides, padding_l, padding_r, dilates,
          groups(), src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
          op_attr, dnnl::algorithm::deconvolution_direct, dnnl::prop_kind::forward_inference,
          ideep::u8s8, ideep::engine::cpu_engine());
    } else {
      ideep::convolution_forward::compute_v2(
          src, weights, dst_dims, dst,
          strides, dilates, padding_l, padding_r, groups(),
          src_scales, weights_scales, dst_scales, src_zero_points, dst_zero_points,
          op_attr, dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
          ideep::u8s8, ideep::engine::cpu_engine());
    }
  }
  return output;
}

template at::Tensor PackedConvWeightsMkldnn<2>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsMkldnn<2>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsMkldnn<3>::apply(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightsMkldnn<3>::apply_relu(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point);

#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace {

template <int kSpatialDim, bool kReluFused>
class QConvInt8Mkldnn final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(
        act.ndimension() == kSpatialDim + 2,
        "Inputs are expected to have ", kSpatialDim + 2, " dimensions");
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

template <bool kReluFused>
class QConv1dInt8Mkldnn final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(
        act.ndimension() == 3,
        "Inputs are expected to have 3 dimensions");
    at::Tensor output;
    // N, C, L -> N, C, 1, L
    act = act.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    if (kReluFused) {
      output = packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      output = packed_weight->apply(act, output_scale, output_zero_point);
    }
    // N, C, 1, L -> N, C, L
    return output.squeeze_(quant_utils::kConv1dSqueezeDim + 2);
  }
};

// kernel for maintaining backward compatibility
template <int kSpatialDim, bool kReluFused>
class QConvInt8ForBCMkldnn final {
 public:
  static Tensor run(
      Tensor act,
      const c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>>& packed_weight,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(
        act.ndimension() == kSpatialDim + 2,
        "Inputs are expected to have ", kSpatialDim + 2, " dimensions");
    if (kReluFused) {
      TORCH_WARN_ONCE(
          "Arguments [stride, padding, dilation, groups] in ops.quantized.conv"
          + c10::to_string(kSpatialDim) + "d_relu, " +
          "have been removed, please update your model to remove these arguments.");
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      TORCH_WARN_ONCE(
          "Arguments [stride, padding, dilation, groups] in ops.quantized.conv"
          + c10::to_string(kSpatialDim) + "d, " +
          "have been removed, please update your model to remove these arguments.");
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_mkldnn"),          QConv1dInt8Mkldnn<false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_relu_mkldnn"),     QConv1dInt8Mkldnn<true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_mkldnn.new"),      QConvInt8Mkldnn<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu_mkldnn.new"), QConvInt8Mkldnn<2, true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_mkldnn.new"),      QConvInt8Mkldnn<3, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu_mkldnn.new"), QConvInt8Mkldnn<3, true>::run);
  // for backward compatibility
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_mkldnn"), QConvInt8ForBCMkldnn<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_relu_mkldnn"), QConvInt8ForBCMkldnn<2, true>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_mkldnn"), QConvInt8ForBCMkldnn<3, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_relu_mkldnn"), QConvInt8ForBCMkldnn<3, true>::run);

  // transpose
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_mkldnn"),  QConv1dInt8Mkldnn<false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_mkldnn"),  QConvInt8Mkldnn<2, false>::run);
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_mkldnn"),  QConvInt8Mkldnn<3, false>::run);
}

} // namespace
} // namespace native
} // namespace at
