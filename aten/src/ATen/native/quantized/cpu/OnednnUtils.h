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
  Gelu
};


struct PackedLinearWeightsOnednn : public LinearPackedParamsBase {
  PackedLinearWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      std::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      std::optional<at::Tensor> orig_bias)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        orig_weight_(std::move(orig_weight)),
        orig_bias_(std::move(orig_bias)) {
    cache_initialized_flag = std::make_unique<c10::once_flag>();
  }
  std::unique_ptr<ideep::tensor> weight_;
  std::optional<ideep::tensor> bias_;
  at::Tensor orig_weight_;
  std::optional<at::Tensor> orig_bias_;

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

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  std::optional<at::Tensor> bias() override {
    return orig_bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias);

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
      std::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      std::optional<at::Tensor> orig_bias,
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
  std::optional<ideep::tensor> bias_;
  at::Tensor orig_weight_;
  std::optional<at::Tensor> orig_bias_;
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

  std::tuple<at::Tensor, std::optional<at::Tensor>> unpack() override;

  static c10::intrusive_ptr<ConvPackedParamsBase<kSpatialDim>> prepack(
      at::Tensor weight,
      std::optional<at::Tensor> bias,
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
      const std::optional<at::Tensor>& accum,
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

inline ideep::attr_t create_attr_by_post_op(
    const c10::string_view& binary_post_op,
    double binary_alpha,
    double input1_scale,
    int64_t input1_zero_point,
    const ideep::tensor::desc& input1_desc,
    const c10::string_view& unary_post_op,
    const torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    const c10::string_view& unary_post_op_algorithm) {
  using ideep::tensor;
  if (binary_post_op == "none") {
    if (unary_post_op == "relu") {
      return ideep::attr_t::fuse_relu();
    } else if (unary_post_op == "leaky_relu") {
      TORCH_CHECK(
          unary_post_op_args.size() == 1,
          "onednn qlinear: expect one argument for post op leaky_relu but got ", unary_post_op_args.size(), " args");
      auto alpha = unary_post_op_args[0].value().to<float>();
      return ideep::attr_t::fuse_relu_v2(alpha);
    } else if (unary_post_op == "tanh") {
      return ideep::attr_t::fuse_tanh();
    } else if (unary_post_op == "gelu") {
      TORCH_CHECK(
          unary_post_op_algorithm == "none" || unary_post_op_algorithm == "tanh",
          "onednn qlinear: algorithm for post op gelu must be none or tanh but got ", unary_post_op_algorithm);
      auto post_algorithm = unary_post_op_algorithm == "none" ?
        dnnl::algorithm::eltwise_gelu_erf :
        dnnl::algorithm::eltwise_gelu_tanh;
      return ideep::attr_t::fuse_gelu_v2(0.f, 0.f, post_algorithm);
    } else if (unary_post_op == "hardtanh") {
      TORCH_CHECK(
          unary_post_op_args.size() == 2 &&
              unary_post_op_args[0].has_value() &&
              unary_post_op_args[1].has_value(),
          "hardtanh is expected to have two scalar input: min_val and max_val");
      auto lower_bound_value =
          unary_post_op_args[0].value().to<float>();
      auto upper_bound_value =
          unary_post_op_args[1].value().to<float>();
      return ideep::attr_t::fuse_clamp(lower_bound_value, upper_bound_value);
    } else if (unary_post_op == "hardswish") {
      return ideep::attr_t::fuse_hardswish();
    } else if (unary_post_op == "swish") {
      return ideep::attr_t::fuse_swish();
    } else {
      TORCH_CHECK(
          unary_post_op == "none",
          "onednn qlinear: unsupported unary post op ", unary_post_op);
    }
  } else if (binary_post_op == "sum") {
    if (unary_post_op == "none") {
      return ideep::attr_t::fuse_sum(input1_scale, input1_zero_point);
    } else if (unary_post_op == "relu") {
      return ideep::attr_t::residual_with_sum_zero_point(input1_scale, input1_zero_point);
    } else {
      TORCH_CHECK(
          false,
          "onednn qlinear: unsupported unary post op ", unary_post_op, " with binary post op sum");
    }
  } else if (binary_post_op == "add") {
    if (unary_post_op == "none") {
      return ideep::attr_t::fuse_binary(ideep::algorithm::binary_add, input1_desc);
    } else if (unary_post_op == "relu") {
      ideep::post_ops po;
      po.append_binary(ideep::algorithm::binary_add, input1_desc);
      po.append_eltwise(ideep::algorithm::eltwise_relu, 0, 0);
      return ideep::attr_t::attr_post_ops(po);
    } else {
      TORCH_CHECK(
          false,
          "onednn qlinear: unsupported unary post op ", unary_post_op, " with binary post op add");
    }
  } else {
    TORCH_CHECK(
        false,
        "onednn qlinear: unsupported binary post op ", binary_post_op);
  }
  return ideep::attr_t();
}

// ONEDNN requires symmetric quantization of weight
// Use this util function to check.
inline bool is_weight_symmetric_quant(
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
inline bool should_use_onednn_quant(
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
    std::optional<torch::List<int64_t>> input_shape=c10::nullopt);

#endif // #if AT_MKLDNN_ENABLED()
