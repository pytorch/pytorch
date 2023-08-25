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
    int64_t, // OMP_number_of_threads
    double, // accum_scale
    int64_t>; // accum_zero_point

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

  LinearParams param;

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
};

struct ConvPrimitiveCache : PrimitiveCache {
  ConvPrimitiveCache() {}

  ConvPrimitiveCache(
      const PrimitiveCacheKey& key,
      const ConvParams& params) {
    this->key = key;
    this->params = params;
  }

  ConvParams params;

  ConvParams& get_params() {
    return params;
  }
};

struct DeconvPrimitiveCache : PrimitiveCache {
  DeconvPrimitiveCache() {}

  DeconvPrimitiveCache(
      const PrimitiveCacheKey& key,
      const DeconvParams& params) {
    this->key = key;
    this->params = params;
  }

  DeconvParams params;

  DeconvParams& get_params() {
    return params;
  }
};

enum PostOps {
  NoPostOp,
  Relu,
  LeakyRelu,
  Tanh,
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

  at::Tensor apply_tanh(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);

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

  at::Tensor apply_add(
      const at::Tensor& input,
      const at::Tensor& accum,
      double output_scale,
      int64_t output_zero_point);

  at::Tensor apply_add_relu(
      const at::Tensor& input,
      const at::Tensor& accum,
      double output_scale,
      int64_t output_zero_point);

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
      const c10::optional<at::Tensor>& accum,
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

// When qengine is x86, use this util func to check if onednn kernel
// is preferred than fbgemm's to get better performance.
static bool should_use_onednn_quant(
    const at::Tensor& weight,
    bool is_transposed_conv,
    int groups,
    torch::List<int64_t> output_padding) {
  // Performance of onednn is only validated on Linux right now.
  // Also, the heuristics for dispatching are based on perf data on Linux.
  // So, for x86 qengine, we always use fbgemm kernels if OS is not Linux.
  // TODO Support more OSs.
#if !defined(__linux__)
  return false;
#else
  bool vnni_available = cpuinfo_has_x86_avx512vnni();
  bool w_sym_quant =
      is_weight_symmetric_quant(weight, is_transposed_conv);
  bool opad_all_zero =
      std::all_of(output_padding.begin(), output_padding.end(), [](int i) { return i==0; });
  return vnni_available && (groups <= 100) && w_sym_quant && opad_all_zero;
#endif
}

} // onednn_utils

at::Tensor _qconv_prepack_onednn(
    at::Tensor weight, // from CPU backend instead of QuantizedCPU
    at::Tensor weight_scales, // Weight zero points must be 0 for onednn
    double input_scale,
    int64_t input_zero_point,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    c10::optional<torch::List<int64_t>> input_shape=c10::nullopt);

static at::Tensor _quantized_convolution_onednn(
    at::Tensor act, // contains quantized values but not QTensor
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight, // MKLDNN tensor with quantized values
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    c10::optional<at::Tensor> bias, // Bias is packed if not None
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    bool transposed,
    int64_t groups,
    double inv_output_scale,
    int64_t output_zero_point,
    c10::optional<at::Tensor> accum=c10::nullopt, // accum to fused with conv add
    double accum_scale=1.0,
    int64_t accum_zero_point=0,
    bool fp32_output=false,
    c10::optional<c10::string_view> binary_attr=c10::nullopt,
    c10::optional<at::Scalar> binary_alpha=c10::nullopt,
    c10::optional<c10::string_view> unary_attr=c10::nullopt,
    torch::List<c10::optional<at::Scalar>> unary_scalars=torch::List<c10::optional<at::Scalar>>(),
    c10::optional<c10::string_view> unary_algorithm=c10::nullopt);

#endif // #if AT_MKLDNN_ENABLED()
