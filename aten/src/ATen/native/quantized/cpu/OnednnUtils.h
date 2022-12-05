#pragma once

#include <ATen/Config.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/Tensor.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ideep.hpp>
#include <cpuinfo.h>

#include <c10/util/CallOnce.h>

using PrimitiveCacheKey = std::tuple<
    double, // input_scale
    int64_t, // input_zero_point
    std::vector<int64_t>, // input_shape
    double, // output_scale
    int64_t, // output_zero_point
    int64_t>; // OMP_number_of_threads

enum CacheKeyIndex {
  InputScale,
  InputZeroPoint,
  InputShape,
  OutputScale,
  OutputZeroPoint,
  NumOfThreads,
};

// Base class of primitive cache
struct PrimitiveCache {
  PrimitiveCacheKey key;

  bool hit(const PrimitiveCacheKey& key) {
    return this->key == key;
  }
};

using LinearParams = ideep::matmul_forward_params;
using Conv = dnnl::convolution_forward;
using ConvDesc = dnnl::convolution_forward::primitive_desc;
using ConvParams = ideep::convolution_forward_params;
using Deconv = dnnl::deconvolution_forward;
using DeconvDesc = dnnl::deconvolution_forward::primitive_desc;
using DeconvParams = ideep::deconv_forward_params;

struct LinearPrimitiveCache : PrimitiveCache {
  LinearPrimitiveCache() {}

  LinearPrimitiveCache(
      const PrimitiveCacheKey& key,
      const LinearParams& param) {
    this->key = key;
    this->param = param;
  }

  LinearPrimitiveCache(
      const PrimitiveCacheKey& key,
      const LinearParams& param,
      const ideep::tensor& bias) {
    this->key = key;
    this->param = param;
    if (!bias.is_empty()) {
      expected_bias =
          bias.reorder_if_differ_in(param.pd.bias_desc(), param.bias_attr);
    }
  }

  LinearParams param;
  ideep::tensor expected_bias;

  // For dynamic qlinear, scale and zero point
  // are set at execution time. So we only need to compare
  // the rest part of key.
  bool hit_dynamic(const PrimitiveCacheKey& new_key) {
    auto cached_input_shape = std::get<InputShape>(this->key);
    auto new_input_shape = std::get<InputShape>(new_key);
    return (
        cached_input_shape == new_input_shape &&
        std::get<NumOfThreads>(this->key) == std::get<NumOfThreads>(new_key));
  }

  LinearParams& get_param() {
    return param;
  }

  ideep::tensor& get_expected_bias() {
    return expected_bias;
  }
};

struct ConvPrimitiveCache : PrimitiveCache {
  ConvPrimitiveCache() {}

  ConvPrimitiveCache(const PrimitiveCacheKey& key,
                     const ConvDesc& conv_desc,
                     const ideep::tensor& bias,
                     const ideep::attr_t bias_attr) {
    this->key = key;
    this->primitive_desc = conv_desc;
    this->primitive = Conv(this->primitive_desc);
    // Construct tensor of input zero point
    ideep::tensor::desc input_zp_desc = {{1}, ideep::data_type::s32, {1}};
    this->input_zp_tensor.init(input_zp_desc, ideep::engine::cpu_engine());
    auto zp_data_ptr = reinterpret_cast<int32_t *>(this->input_zp_tensor.get_data_handle());
    zp_data_ptr[0] = std::get<InputZeroPoint>(key);
    // Construct expected bias
    this->expected_bias = bias.reorder_if_differ_in(conv_desc.bias_desc(), bias_attr);
  }

  ConvDesc primitive_desc;
  Conv primitive;
  ideep::tensor input_zp_tensor;
  ideep::tensor expected_bias;

  inline ConvDesc& get_primitive_desc() {
    return primitive_desc;
  }

  inline Conv& get_primitive() {
    return primitive;
  }

  inline ideep::tensor& get_src_zp_tensor() {
    return input_zp_tensor;
  }

  inline ideep::tensor& get_bias() {
    return expected_bias;
  }
};

struct DeconvPrimitiveCache : PrimitiveCache {
  DeconvPrimitiveCache() {}

  DeconvPrimitiveCache(const PrimitiveCacheKey& key,
                       const DeconvDesc& deconv_desc,
                       const ideep::tensor& bias,
                       const ideep::attr_t bias_attr,
                       const ideep::tensor& input_zero_point) {
    this->key = key;
    this->primitive_desc = deconv_desc;
    this->primitive = Deconv(this->primitive_desc);
    this->input_zp_tensor = std::move(input_zero_point);
    // Construct expected bias
    this->expected_bias = bias.reorder_if_differ_in(deconv_desc.bias_desc(), bias_attr);
  }

  DeconvDesc primitive_desc;
  Deconv primitive;
  ideep::tensor input_zp_tensor;
  ideep::tensor expected_bias;

  inline DeconvDesc& get_primitive_desc() {
    return primitive_desc;
  }

  inline Deconv& get_primitive() {
    return primitive;
  }

  inline ideep::tensor& get_src_zp_tensor() {
    return input_zp_tensor;
  }

  inline ideep::tensor& get_bias() {
    return expected_bias;
  }
};

enum PostOps {
  NoPostOp,
  Relu,
  LeakyRelu,
};

struct PackedLinearWeightsOnednn : public LinearPackedParamsBase {
  PackedLinearWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      c10::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      c10::optional<at::Tensor> orig_bias)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        orig_weight_(std::move(orig_weight)),
        orig_bias_(std::move(orig_bias)) {
    cache_initialized_flag = std::make_unique<c10::once_flag>();
  }
  std::unique_ptr<ideep::tensor> weight_;
  c10::optional<ideep::tensor> bias_;
  at::Tensor orig_weight_;
  c10::optional<at::Tensor> orig_bias_;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range=false) override;
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range=false) override;

  at::Tensor apply_leaky_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point,
      double negative_slope);

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  c10::optional<at::Tensor> bias() override {
    return orig_bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias);

 private:
  LinearPrimitiveCache prim_cache;
  std::unique_ptr<c10::once_flag> cache_initialized_flag;

  template <PostOps post_op>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point,
      torch::List<at::Scalar> post_op_args = torch::List<at::Scalar>());

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range=false);

  LinearPrimitiveCache& get_cache() {
    return prim_cache;
  }
};

template <int kSpatialDim = 2>
struct PackedConvWeightsOnednn : public ConvPackedParamsBase<kSpatialDim> {
  PackedConvWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      c10::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      c10::optional<at::Tensor> orig_bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      uint8_t transpose)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        orig_weight_(std::move(orig_weight)),
        orig_bias_(std::move(orig_bias)),
        stride_(std::move(stride)),
        padding_(std::move(padding)),
        output_padding_(std::move(output_padding)),
        dilation_(std::move(dilation)),
        groups_(groups),
        transpose_(transpose) {
    cache_initialized_flag = std::make_unique<c10::once_flag>();
  }

  std::unique_ptr<ideep::tensor> weight_;
  c10::optional<ideep::tensor> bias_;
  at::Tensor orig_weight_;
  c10::optional<at::Tensor> orig_bias_;
  torch::List<int64_t> stride_;
  torch::List<int64_t> padding_;
  torch::List<int64_t> output_padding_;
  torch::List<int64_t> dilation_;
  int64_t groups_;
  uint8_t transpose_;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(
      const at::Tensor& input,
      bool reduce_range) override;

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> output_padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      bool transpose);

  torch::List<int64_t> stride() const override {
    return stride_;
  }

  torch::List<int64_t> padding() const override {
    return padding_;
  }

  torch::List<int64_t> output_padding() const override {
    return output_padding_;
  }

  torch::List<int64_t> dilation() const override {
    return dilation_;
  }

  int64_t groups() const override {
    return groups_;
  }

  bool transpose() const override {
    return (bool)transpose_;
  }

 private:
  ConvPrimitiveCache conv_prim_cache;
  DeconvPrimitiveCache deconv_prim_cache;
  std::unique_ptr<c10::once_flag> cache_initialized_flag;

  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);

  ConvPrimitiveCache& get_conv_cache() {
    assert(!transpose());
    return conv_prim_cache;
  }

  DeconvPrimitiveCache& get_deconv_cache() {
    assert(transpose());
    return deconv_prim_cache;
  }
};

namespace onednn_utils {

// Try to reorder tensor to expected desc at runtime
// Do it in a `try...catch...` manner to avoid oneDNN's errors
// TODO: Move it to third_party/ideep
static void try_reorder(
    ideep::tensor& t,
    const ideep::tensor::desc&& desc,
    ideep::scale_t scales) {
  if (t.get_desc() != desc) {
    try {
      t = t.reorder_if_differ_in(desc);
    } catch (...) {
      ideep::tensor&& plain = t.to_public(nullptr, t.get_data_type());
      t = plain.reorder_if_differ_in(desc);
    }
    t.set_scale(scales);
  }
}

// ONEDNN requires symmetric quantization of weight
// Use this util function to check.
static bool is_weight_symmetric_quant(
      const at::Tensor& weight,
      bool is_transposed_conv) {
  bool is_symmetric = true;
  const auto qtype = weight.qscheme();
  if (qtype == c10::kPerTensorAffine) {
    is_symmetric &= (weight.q_zero_point() == 0);
  } else if (qtype == c10::kPerChannelAffine) {
    if (is_transposed_conv) {
      // This case is currently not supported in PyTorch
      // but we do not want to raise an error in this util function.
      is_symmetric = false;
    } else {
      auto output_channels = weight.size(0);
      for (int i = 0; i < output_channels; ++i) {
        auto zp = weight.q_per_channel_zero_points()[i].item<int32_t>();
        is_symmetric &= (zp == 0);
      }
    }
  } else {
    // This case is currently not supported in PyTorch
      // but we do not want to raise an error in this util function.
    is_symmetric = false;
  }
  return is_symmetric;
}

// Check if onednn should be used w.r.t fbgemm
static bool should_use_onednn_quant(
    const at::Tensor& weight,
    bool is_transposed_conv,
    int groups,
    torch::List<int64_t> output_padding) {
  bool vnni_available = cpuinfo_has_x86_avx512vnni();
  bool w_sym_quant =
      is_weight_symmetric_quant(weight, is_transposed_conv);
  bool opad_all_zero =
      std::all_of(output_padding.begin(), output_padding.end(), [](int i) { return i==0; });
  return vnni_available && (groups <= 100) && w_sym_quant && opad_all_zero;
}

} // onednn_utils

#endif // #if AT_MKLDNN_ENABLED()
