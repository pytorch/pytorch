#include <array>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/mkldnn/quantization/Utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/library.h>

#include <c10/util/irange.h>

#if AT_MKLDNN_ENABLED()
template <int kSpatialDim>
c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> PackedConvWeightsMkldnn<
    kSpatialDim>::
    prepack(
        at::Tensor weight,
        c10::optional<at::Tensor> bias,
        torch::List<int64_t> stride,
        torch::List<int64_t> padding,
        torch::List<int64_t> output_padding,
        torch::List<int64_t> dilation,
        int64_t groups,
        bool transpose) {
  TORCH_CHECK(
      weight.ndimension() == kSpatialDim + 2,
      "Weights are expected to have ", kSpatialDim + 2, " dimensions");
  TORCH_CHECK(
      stride.size() == kSpatialDim,
      "stride should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");
  TORCH_CHECK(
      padding.size() == kSpatialDim,
      "Specify front/top/left padding only. "
      "end/bottom/right padding assumed to be equal to front/top/left");
  TORCH_CHECK(
      !transpose || output_padding.size() == kSpatialDim,
      "quantized::conv_prepack: Specify top/left output padding "
      "only. bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(
      dilation.size() == kSpatialDim,
      "dilation should contain ", kSpatialDim, " elements for ",
      kSpatialDim, "D convolution.");
  TORCH_CHECK(
      !transpose || std::all_of(output_padding.begin(), output_padding.end(), [](int i) { return i==0; }),
      "quantized::conv_prepack: MKLDNN only supports zero output_padding.");
  TORCH_CHECK(weight.scalar_type() == c10::ScalarType::QInt8,
      "Data type of weight should be Qint8.");

  // Weight
  // Format: [OC IC//group KH KW] for conv; [IC OC//group KH KW] for deconv
  auto dims = weight.sizes().vec();
  auto strides = stride.vec();
  auto padding_l = padding.vec();
  auto padding_r = padding.vec();
  auto dilates = dilation.vec();
  auto op_attr = ideep::attr_t();
  std::vector<int32_t> wgt_zero_points;
  ideep::scale_t wgt_scales;
  const int output_channels = transpose ? weight.size(1) * groups
                                        : weight.size(0);
  const auto qtype = weight.qscheme();
  if (qtype == c10::kPerTensorAffine) {
    TORCH_CHECK(
        weight.q_zero_point()==0,
        "quantized::qconv_prepack: MKLDNN only supports symmetric quantization of weight,"
        " whose zero point must be 0.");
    wgt_zero_points = std::vector<int32_t>(1, weight.q_zero_point());
    wgt_scales = ideep::scale_t(1, 1.0/weight.q_scale()); // Scales of MKLDNN and PyTorch are reciprocal
  } else if (qtype == c10::kPerChannelAffine) {
    TORCH_CHECK(
        !transpose,
        "Per Channel Quantization is currently disabled for transposed conv");
    wgt_zero_points.resize(output_channels);
    wgt_scales.resize(output_channels);
    for (int i = 0; i < output_channels; ++i) {
      wgt_zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
      TORCH_CHECK(
          wgt_zero_points[i]==0,
          "quantized::qconv_prepack: MKLDNN only supports symmetric quantization of weight,"
          " whose zero point must be 0.");
      wgt_scales[i] = 1.0f / weight.q_per_channel_scales()[i].item<float>(); // Scales of MKLDNN and PyTorch are reciprocal
    }
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  // Set runtime src zero point
  auto src_zero_point = {DNNL_RUNTIME_S32_VAL};
  op_attr.set_zero_points(DNNL_ARG_SRC,
                          ideep::utils::tensor_zp_mask(src_zero_point.size()),
                          src_zero_point);
  at::Tensor weight_copy;
  ideep::tensor::desc w_desc;
  ideep::dims dims_iohw, dims_giohw;
  ideep::tag w_tag = ideep::tag::any;
  const bool with_groups = groups > 1;
  if (transpose) {
    w_desc = ideep::convolution_transpose_forward::expected_weights_desc(
        dims, dnnl::memory::data_type::s8,
        strides, padding_l, padding_r, dilates, groups,
        dnnl::algorithm::deconvolution_direct, dnnl::prop_kind::forward_inference,
        ideep::dims(), op_attr);
    // convolution_transpose_forward::expected_weights_desc() gives format [i, o, ...],
    // but MKLDNN requires [o, i, ...] for computation
    dims_iohw = w_desc.get_dims();
    dims_giohw = with_groups ? ideep::utils::group_dims(dims_iohw, groups) : dims_iohw;
    std::vector<int64_t> perms(dims_giohw.size(), 0); // for permutation of weight
    std::iota(perms.begin(), perms.end(), 0);
    w_desc = w_desc.transpose(with_groups, with_groups + 1);
    std::swap(perms[with_groups], perms[with_groups + 1]);
    weight_copy = weight.reshape(dims_giohw).permute(c10::IntArrayRef(perms)).clone();
  } else {
    w_desc = ideep::convolution_forward::expected_weights_desc(
        dims, dnnl::memory::data_type::s8,
        strides, padding_l, padding_r, dilates, groups,
        dnnl::algorithm::convolution_direct, dnnl::prop_kind::forward_inference,
        dnnl::memory::data_type::u8, ideep::dims(), op_attr);
    weight_copy = weight.clone();
  }
  if (with_groups) {
    w_tag = kSpatialDim == 2 ? ideep::tag::goihw : ideep::tag::goidhw;
  } else {
    w_tag = kSpatialDim == 2 ? ideep::tag::oihw : ideep::tag::oidhw;
  }
  ideep::dims w_dims = with_groups ? ideep::utils::group_dims(w_desc.get_dims(), groups)
                                   : w_desc.get_dims();
  ideep::tensor wgt = ideep::tensor(
      ideep::tensor::desc({w_dims, dnnl::memory::data_type::s8, w_tag}, groups),
      weight_copy.data_ptr());
  wgt.set_scale(wgt_scales); // Scales are needed for feed_from().
  ideep::tensor exp_wgt;
  exp_wgt.init(w_desc);
  exp_wgt.set_scale(wgt_scales); // Also for feed_from()
  exp_wgt.feed_from(wgt, transpose); // expect wgt to be in [OC IC KH KW] format
  ideep::tensor * packed_weight_p = new ideep::tensor(exp_wgt);
  packed_weight_p->set_scale(wgt_scales);
  packed_weight_p->set_zero_point(wgt_zero_points);
  std::unique_ptr<ideep::tensor> weight_ptr(packed_weight_p);
  // Bias
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    auto bias_desc = ideep::tensor::desc(bias.value().sizes().vec(), dnnl::memory::data_type::f32);
    ideep::tensor packed_bias;
    packed_bias.init(bias_desc, bias.value().data_ptr());
    mkldnn_bias = c10::optional<ideep::tensor>(packed_bias);
  }
  auto ret_ptr = c10::make_intrusive<PackedConvWeightsMkldnn<kSpatialDim>>(
      PackedConvWeightsMkldnn<kSpatialDim>{
        std::move(weight_ptr),
        mkldnn_bias,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        transpose
      });
  return ret_ptr;
}

template struct PackedConvWeightsMkldnn<2>;
template struct PackedConvWeightsMkldnn<3>;
#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace {

template <int kSpatialDim = 2>
class QConvPackWeightInt8Mkldnn final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_conv(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    torch::List<int64_t> output_padding;
    output_padding.reserve(kSpatialDim);
    for (int idx = 0; idx < kSpatialDim; ++idx) {
      output_padding.push_back((int64_t)0);
    }
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> run_deconv(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/true);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> _run(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    auto& ctx = at::globalContext();

#if AT_MKLDNN_ENABLED()
    return PackedConvWeightsMkldnn<kSpatialDim>::prepack(
        weight, bias, stride, padding, output_padding, dilation, groups,
        transpose);
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv2d_prepack ",
        toString(ctx.qEngine()));
  }
};



class QConv1dPackWeightInt8Mkldnn final {
 public:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_conv(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    const torch::List<int64_t> output_padding({0});
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/false);
  }

  static c10::intrusive_ptr<ConvPackedParamsBase<2>> run_deconv(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    return _run(weight, bias, stride, padding, output_padding, dilation, groups,
                /*transpose=*/true);
  }

 private:
  static c10::intrusive_ptr<ConvPackedParamsBase<2>> _run(
      Tensor weight,
      c10::optional<Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose) {
    auto& ctx = at::globalContext();
    if (weight.dim() == 3) {
      weight = weight.unsqueeze(quant_utils::kConv1dSqueezeDim + 2);
    }
    stride = quant_utils::MakeArgForConv1d(stride, 1);
    padding = quant_utils::MakeArgForConv1d(padding, 0);
    output_padding = quant_utils::MakeArgForConv1d(output_padding, 0);
    dilation = quant_utils::MakeArgForConv1d(dilation, 1);
#if AT_MKLDNN_ENABLED()
    return PackedConvWeightsMkldnn<2>::prepack(
        weight, bias, stride, padding, output_padding, dilation, groups,
        transpose);
#endif

    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::conv1d_prepack ",
        toString(ctx.qEngine()));
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  // Conv
  // conv_prepack is deprecated, please use conv2d_prepack for 2D conv.
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<2>::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv1d_prepack_mkldnn"), TORCH_FN(QConv1dPackWeightInt8Mkldnn::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv2d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<2>::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv3d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<3>::run_conv));
  // ConvTranspose
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose1d_prepack_mkldnn"), TORCH_FN(QConv1dPackWeightInt8Mkldnn::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose2d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<2>::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("quantized::conv_transpose3d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<3>::run_deconv));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  // Conv
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv2d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<2>::run_conv));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv3d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<3>::run_conv));
  // ConvTranspose
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose1d_prepack_mkldnn"), TORCH_FN(QConv1dPackWeightInt8Mkldnn::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose2d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<2>::run_deconv));
  m.impl(TORCH_SELECTIVE_NAME("_quantized::conv_transpose3d_prepack_mkldnn"), TORCH_FN(QConvPackWeightInt8Mkldnn<3>::run_deconv));
}

} // namespace
} // namespace native
} // namespace at
