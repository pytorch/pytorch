#pragma once

#ifdef USE_FBGEMM
#include <ATen/Config.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>

struct FBGEMM_API PackedWeightQmkldnn {
  std::unique_ptr<ideep::tensor> w;
  c10::optional<at::Tensor> bias;
  std::vector<float> w_scale;
  std::vector<int32_t> w_zp;
  c10::QScheme q_scheme;
};

static at::Tensor mkldnn_linear_prepack(
    at::Tensor& weight,
    c10::optional<at::Tensor> bias) {
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack should be 2-dimensional.");
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

  std::vector<int32_t> weight_zero_points(1, 0);
  if (qtype == c10::kPerTensorAffine) {
    weight_zero_points[0] = static_cast<int32_t>(weight_contig.q_zero_point());
  } else if (qtype == c10::kPerChannelAffine) {
    weight_zero_points.resize(N, 0);
    for (int i = 0; i < N; ++i) {
      weight_zero_points[i] =
          weight.q_per_channel_zero_points()[i].item<int32_t>();
    }
  }
  std::vector<float> weight_scales(1, 0.0);
  if (qtype == c10::kPerTensorAffine) {
    weight_scales[0] = weight.q_scale();
  } else if (qtype == c10::kPerChannelAffine) {
    weight_scales.resize(N, 0.0);
    for (int i = 0; i < N; ++i) {
      weight_scales[i] = weight.q_per_channel_scales()[i].item<float>();
    }
  }

  auto scale_ = at::native::ConvertScales(weight_scales);
  int8_t* weight_ptr = reinterpret_cast<int8_t*>(weight_contig.data_ptr());
  ideep::tensor weight_;
  weight_.init(
      weight_contig.sizes().vec(), ideep::tensor::data_type::s8, weight_ptr);
  weight_.set_scale(scale_);
  auto desc =
      ideep::matmul_forward::expected_weights_desc(
          weight_.get_dims(),
          ideep::tensor::data_type::s8,
          ideep::tensor::data_type::u8);
  ideep::tensor output;
  if (weight_.get_desc() != desc) {
    output.init(desc);
    output.set_scale(scale_);
    output.feed_from(weight_);
  } else {
    output = weight_;
  }

  auto ret_ptr = std::make_unique<PackedWeightQmkldnn>(
      PackedWeightQmkldnn{
          std::make_unique<ideep::tensor>(output),
          bias_contig,
          weight_scales,
          weight_zero_points,
          qtype});
  return at::cpp_custom_type_hack::create(std::move(ret_ptr), weight.options());
}

static std::tuple<at::Tensor, c10::optional<at::Tensor>> mkldnn_linear_unpack(
    at::Tensor packed_weight) {
  auto& pack_ptr =
      at::cpp_custom_type_hack::cast<PackedWeightQmkldnn>(
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
        device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), device(c10::kCPU).dtype(c10::kInt));

    weight_origin = at::_empty_per_channel_affine_quantized(
        {N, K},
        scales,
        zero_points,
        0, // The output channel axis is 0
        device(c10::kCPU).dtype(c10::kQInt8));
  }

  int8_t* weight_ptr_int8 =
      reinterpret_cast<int8_t*>(weight_origin.data_ptr<c10::qint8>());

  packB_mkldnn->to_public(weight_ptr_int8,
                          /*dst_type=*/ideep::data_type::undef);

  return std::tuple<at::Tensor, c10::optional<at::Tensor>>(weight_origin, bias);
}
#endif // AT_MKLDNN_ENABLED()
#endif // USE_FBGEMM
