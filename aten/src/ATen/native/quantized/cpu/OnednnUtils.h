#pragma once

#include <ATen/Config.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/Tensor.h>
#include <ATen/native/quantized/packed_params.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

struct PackedLinearWeightsOnednn : public LinearPackedParamsBase {
  PackedLinearWeightsOnednn(
      std::unique_ptr<ideep::tensor> weight,
      c10::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      c10::optional<at::Tensor> orig_bias)
      : weight_(std::move(weight)),
        bias_(std::move(bias)),
        orig_weight_(std::move(orig_weight)),
        orig_bias_(std::move(orig_bias)) {}
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

  std::tuple<at::Tensor, c10::optional<at::Tensor>> unpack() override;

  c10::optional<at::Tensor> bias() override {
    return orig_bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      at::Tensor weight,
      c10::optional<at::Tensor> bias);

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range=false);
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
    transpose_(transpose) {}

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
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
};

#endif // #if AT_MKLDNN_ENABLED()
