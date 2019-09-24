#pragma once

#ifdef USE_FBGEMM
#include <ATen/Config.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

struct FBGEMM_API PackedLinearWeightQmkldnn {
  std::unique_ptr<ideep::tensor> w;
  c10::optional<at::Tensor> bias;
  std::vector<double> w_scale;
  std::vector<int64_t> w_zp;
  c10::QScheme q_scheme;
};

struct FBGEMM_API PackedConvWeightQmkldnn {
  std::unique_ptr<ideep::tensor> w;
  c10::optional<at::Tensor> bias;
  std::vector<double> w_scale;
  std::vector<int64_t> w_zp;
  c10::QScheme q_scheme;
};

static at::Tensor mkldnn_linear_prepack(
    at::Tensor& weight,
    c10::optional<at::Tensor> bias) {
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack should"
      " be 2-dimensional.");
  auto N = weight.size(0);
  // TODO: contiguous is called for further JIT optimizations.
  auto weight_contig = weight.contiguous();
  const auto qtype = weight.qscheme();

  c10::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == N,
        "bias should have N elements: " + std::to_string(N));
    bias_contig = bias->contiguous();
  }

  // This is a terrible hack to emulate what VariableType is doing
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  auto quantizer = at::get_qtensorimpl(weight_contig)->quantizer();

  std::vector<int64_t> weight_zero_points(1, 0);
  if (qtype == c10::kPerTensorAffine) {
    weight_zero_points[0] = weight_contig.q_zero_point();
  } else if (qtype == c10::kPerChannelAffine) {
    weight_zero_points =
        static_cast<at::PerChannelAffineQuantizer*>(quantizer.get())
            ->zero_points();
  }
  std::vector<double> weight_scales(1, 0.0);
  if (qtype == c10::kPerTensorAffine) {
    weight_scales[0] = weight.q_scale();
  } else if (qtype == c10::kPerChannelAffine) {
    weight_scales =
        static_cast<at::PerChannelAffineQuantizer*>(quantizer.get())->scales();
  }

  auto scale_ = at::native::ConvertScales(weight_scales);
  int8_t* weight_ptr = reinterpret_cast<int8_t*>(weight_contig.data_ptr());
  ideep::tensor weight_;
  weight_.init(
      {{weight_contig.sizes().cbegin(), weight_contig.sizes().cend()},
       ideep::tensor::data_type::s8},
      weight_ptr);
  weight_ = weight_.as_weights();
  weight_.set_scale(scale_);
  ideep::tensor::descriptor desc =
      ideep::inner_product_forward::expected_weights_descriptor(
          weight_.get_dims(),
          ideep::tensor::data_type::s8,
          ideep::tensor::data_type::u8);
  ideep::tensor output;
  if (weight_.get_descriptor() != desc) {
    output.init<at::native::AllocForMKLDNN>(desc);
    output.set_scale(scale_);
    output.feed_from(weight_);
  } else {
    output = weight_;
  }

  auto ret_ptr = c10::guts::make_unique<PackedLinearWeightQmkldnn>(
      PackedLinearWeightQmkldnn{
          c10::guts::make_unique<ideep::tensor>(output),
          bias_contig,
          weight_scales,
          weight_zero_points,
          qtype});
  return at::cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
}

static std::tuple<at::Tensor, c10::optional<at::Tensor>> mkldnn_linear_unpack(
    at::Tensor packed_weight) {
  auto& pack_ptr =
      at::cpp_custom_type_hack::cast<PackedLinearWeightQmkldnn>(
          packed_weight);
  auto packB_mkldnn = reinterpret_cast<ideep::tensor*>(pack_ptr.w.get());
  int64_t N = packB_mkldnn->get_dim(0);
  int64_t K = packB_mkldnn->get_dim(1);
  auto w_scale = pack_ptr.w_scale;
  auto w_zp = pack_ptr.w_zp;
  auto qscheme = pack_ptr.q_scheme;
  auto bias = pack_ptr.bias;
  at::Tensor weight_origin;
  if (qscheme == c10::kPerTensorAffine) {
    weight_origin = at::_empty_affine_quantized(
        {N, K},
        at::device(c10::kCPU).dtype(c10::kQInt8),
        w_scale[0],
        w_zp[0]);
  } else if (qscheme == c10::kPerChannelAffine) {
    auto scales = at::from_blob(
        w_scale.data(),
        w_scale.size(),
        device(c10::kCPU).dtype(c10::kDouble));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kLong));

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales,
        zero_points,
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  packB_mkldnn->to_public_format(weight_ptr_int8);

  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      weight_origin, bias);
}

static at::Tensor mkldnn_conv_prepack(
    at::Tensor& weight,
    c10::optional<at::Tensor>& bias,
    torch::List<int64_t>& stride,
    torch::List<int64_t>& padding,
    torch::List<int64_t>& dilation,
    int64_t groups) {
  TORCH_CHECK(
      weight.ndimension() == 4, "Weights are expected to have 4 dimensions");
  TORCH_CHECK(stride.size() == 2, "2D convolution only");
  TORCH_CHECK(
      padding.size() == 2,
      "Specify top/left padding only. \
    bottom/right padding assumed to be equal to top/left");
  TORCH_CHECK(dilation.size() == 2, "2D convolution only");

  int output_channels = weight.size(0);

  const auto qtype = weight.qscheme();

  auto weight_contig = weight.contiguous();

  // This is a terrible hack to emulate what VariableType is doing
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  auto quantizer = at::get_qtensorimpl(weight_contig)->quantizer();

  std::vector<int64_t> zero_points(1, 0);
  if (qtype == c10::kPerTensorAffine) {
    zero_points[0] = weight_contig.q_zero_point();
  } else if (qtype == c10::kPerChannelAffine) {
    int64_t axis = weight_contig.q_per_channel_axis();
    TORCH_CHECK(
        axis == 0,
        "Only per output channel quantization is supported for the weights");
    zero_points = static_cast<at::PerChannelAffineQuantizer*>(quantizer.get())
                      ->zero_points();
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  std::vector<double> scales(1, 0.0);
  if (qtype == c10::kPerTensorAffine) {
    scales[0] = weight.q_scale();
  } else if (qtype == c10::kPerChannelAffine) {
    scales =
        static_cast<at::PerChannelAffineQuantizer*>(quantizer.get())->scales();
  }

  c10::optional<at::Tensor> bias_contig;
  if (bias.has_value()) {
    at::Tensor bias_vec = bias.value();
    TORCH_CHECK(bias_vec.dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias_vec.size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
    bias_contig = bias->contiguous();
  }

  // packed MKL-DNN weight
  auto scale_ = at::native::ConvertScales(scales);


  int8_t *weight_ptr =
      reinterpret_cast<int8_t*>(weight_contig.data_ptr<c10::qint8>());
  ideep::tensor weight_;
  weight_.init({{weight_contig.sizes().cbegin(), weight_contig.sizes().cend()},
                ideep::tensor::data_type::s8}, weight_ptr);
  weight_ = weight_.as_weights();
  weight_.set_scale(scale_);
  weight_.make_group(groups);

  ideep::tensor::descriptor desc =
      ideep::convolution_forward::expected_weights_descriptor(
          weight_.get_dims(),
          ideep::tensor::data_type::s8,
          {stride.begin(), stride.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          {dilation.begin(), dilation.end()},
          groups,
          ideep::algorithm::convolution_direct,
          ideep::prop_kind::forward_inference,
          ideep::tensor::data_type::u8); // TODO: u8/s8
  ideep::tensor output;
  if (weight_.get_descriptor() != desc) {
    output.init<at::native::AllocForMKLDNN>(desc);
    output.set_scale(scale_);
    output.feed_from(weight_);
  } else {
    output = weight_;
  }

  auto ret_ptr = c10::guts::make_unique<PackedConvWeightQmkldnn>(
      PackedConvWeightQmkldnn{
          c10::guts::make_unique<ideep::tensor>(output),
          bias_contig,
          scales,
          zero_points,
          qtype});
  return at::cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
}

static std::tuple<at::Tensor, c10::optional<at::Tensor>> mkldnn_conv_unpack(
    at::Tensor packed_weights) {
  int64_t output_channels, C_per_G, kernel_h, kernel_w;
  auto& pack_ptr =
      at::cpp_custom_type_hack::cast<PackedConvWeightQmkldnn>(
          packed_weights);
  auto qscheme = pack_ptr.q_scheme;
  auto packB_mkldnn = reinterpret_cast<ideep::tensor*>(pack_ptr.w.get());
  if (packB_mkldnn->ndims() == 5) {
    output_channels = packB_mkldnn->get_dim(0) * packB_mkldnn->get_dim(1);
    C_per_G = packB_mkldnn->get_dim(2);
    kernel_h = packB_mkldnn->get_dim(3);
    kernel_w =packB_mkldnn->get_dim(4);
  } else {
    output_channels = packB_mkldnn->get_dim(0);
    C_per_G = packB_mkldnn->get_dim(1);
    kernel_h = packB_mkldnn->get_dim(2);
    kernel_w =packB_mkldnn->get_dim(3);
  }

  auto w_scale = pack_ptr.w_scale;
  auto w_zp = pack_ptr.w_zp;

  auto bias = pack_ptr.bias;

  at::Tensor unpacked_weights;
  if (qscheme == c10::kPerTensorAffine) {
      unpacked_weights = at::_empty_affine_quantized(
          {output_channels, C_per_G, kernel_h, kernel_w},
          device(c10::kCPU).dtype(c10::kQInt8),
          w_scale[0],
          w_zp[0]);
  } else if (qscheme == c10::kPerChannelAffine) {
    auto scales = at::from_blob(
        w_scale.data(),
        w_scale.size(),
        device(c10::kCPU).dtype(c10::kDouble));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kLong));
      unpacked_weights = at::_empty_per_channel_affine_quantized(
          {output_channels, C_per_G, kernel_h, kernel_w},
          scales,
          zero_points,
          0, /* The output channel axis is 0 */
          device(c10::kCPU).dtype(c10::kQInt8));
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qscheme));
  }

  int8_t* unpacked_weights_p =
      reinterpret_cast<int8_t*>(unpacked_weights.data_ptr<c10::qint8>());

  packB_mkldnn->to_public_format(unpacked_weights_p);
  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(
      unpacked_weights, bias);
}

inline bool use_mkldnn(
    at::Tensor& input,
    int64_t output_zero_point,
    bool relu_fused) {
  // Currently, only u8s8->u8 is support for AQ
  if (!relu_fused || input.scalar_type() != at::kQUInt8 ||
      input.q_zero_point() || output_zero_point)
    return false;
  if (at::globalContext().qEngine() == at::kQMKLDNN)
    return true;
  else
    return false;
}

template <typename T = int32_t>
inline bool is_zeros(at::Tensor zero_points) {
  for (int i = 0; i < zero_points.numel(); i++) {
    if (zero_points[i].item<T>() != 0)
      return false;
  }
  return true;
}

#endif // AT_MKLDNN_ENABLED()
#endif // USE_FBGEMM
