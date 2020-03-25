#pragma once

#ifdef USE_XNNPACK

#include <ATen/core/ivalue.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/Tensor.h>

namespace at {
namespace native {
namespace xnnpack {

using SerializationTypeLinearPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    c10::optional<double>,
    c10::optional<double>>;
using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    c10::optional<double>,
    c10::optional<double>>;

class LinearOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  c10::optional<Tensor> orig_bias_;
  c10::optional<double> output_min_;
  c10::optional<double> output_max_;

 public:
  SerializationTypeLinearPrePack unpack() {
    return std::make_tuple(orig_weight_, orig_bias_, output_min_, output_max_);
  }

  virtual Tensor run(const Tensor& input) = 0;
};

class XNNPackLinearOpContext final : public LinearOpContext {
 private:
  ContextLinear op_context_;

 public:
  XNNPackLinearOpContext(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      ContextLinear&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
  }

  Tensor run(const Tensor& input);

  static c10::intrusive_ptr<LinearOpContext> create_context(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      const c10::optional<double> output_min,
      const c10::optional<double> output_max);
};

class Conv2dOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  c10::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  c10::optional<double> output_min_;
  c10::optional<double> output_max_;

 public:
  SerializationTypeConv2dPrePack unpack() {
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }

  virtual Tensor run(const Tensor& input) = 0;
};

class XNNPackConv2dOpContext final : public Conv2dOpContext {
 private:
  ContextConv2D op_context_;

 public:
  XNNPackConv2dOpContext(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      ContextConv2D&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
  }

  Tensor run(const Tensor& input);

  static c10::intrusive_ptr<Conv2dOpContext> create_context(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      const c10::optional<double> output_min,
      const c10::optional<double> output_max);
};
} // namespace xnnpack

} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
