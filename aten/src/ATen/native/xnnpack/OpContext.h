#pragma once

#ifdef USE_XNNPACK

#include <ATen/core/ivalue.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/Tensor.h>

namespace at::native::xnnpack {

using SerializationTypeLinearPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    c10::optional<Scalar>,
    c10::optional<Scalar>>;
using SerializationTypeConv2dPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    c10::optional<Scalar>,
    c10::optional<Scalar>>;
using SerializationTypeTransposeConv2dPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    c10::optional<Scalar>,
    c10::optional<Scalar>>;



class LinearOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  c10::optional<Tensor> orig_bias_;
  c10::optional<Scalar> output_min_;
  c10::optional<Scalar> output_max_;
  bool orig_weight_and_bias_freed_;

 public:
  SerializationTypeLinearPrePack unpack() {
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    return std::make_tuple(orig_weight_, orig_bias_, output_min_, output_max_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  virtual void free_orig_weight_and_bias() = 0;
};

class XNNPackLinearOpContext final : public LinearOpContext {
 private:
  ContextLinear op_context_;

 public:
  XNNPackLinearOpContext(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      const c10::optional<Scalar>& min,
      const c10::optional<Scalar>& max,
      ContextLinear&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    output_min_ = min;
    output_max_ = max;
    orig_weight_and_bias_freed_ = false;
  }

  Tensor run(const Tensor& input) override;
  void free_orig_weight_and_bias() override;

  static c10::intrusive_ptr<LinearOpContext> create_context(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      const c10::optional<Scalar>& output_min,
      const c10::optional<Scalar>& output_max);
};

class Conv2dOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  c10::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  c10::optional<Scalar> output_min_;
  c10::optional<Scalar> output_max_;
  bool orig_weight_and_bias_freed_;

 public:
  SerializationTypeConv2dPrePack unpack() {
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
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
  virtual void free_orig_weight_and_bias() = 0;
};

class TransposeConv2dOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  c10::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  c10::optional<Scalar> output_min_;
  c10::optional<Scalar> output_max_;
  bool orig_weight_and_bias_freed_;

 public:
  SerializationTypeTransposeConv2dPrePack unpack() {
    TORCH_CHECK(!orig_weight_and_bias_freed_, "Original weight and bias have been freed");
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        output_padding_,
        dilation_,
        groups_,
        output_min_,
        output_max_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  virtual void free_orig_weight_and_bias() = 0;
};

class XNNPackConv2dOpContext final : public Conv2dOpContext {
 private:
  ContextConv2D op_context_;
  // xnnpack convs use indirection buffer.
  // These buffers need setup at runtime and/or when input
  // dims change. If we are running the same model on multiple
  // threads, this can lead to contention where indirection buffer
  // is being accessed and updated at the same time from two different
  // threads.
  std::mutex xnnp_mutex_;

 public:
  XNNPackConv2dOpContext(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      const c10::optional<Scalar>& min,
      const c10::optional<Scalar>& max,
      ContextConv2D&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
    output_min_ = min;
    output_max_ = max;
    orig_weight_and_bias_freed_ = false;
  }

  Tensor run(const Tensor& input) override;
  void free_orig_weight_and_bias() override;

  static c10::intrusive_ptr<Conv2dOpContext> create_context(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      const c10::optional<Scalar>& output_min,
      const c10::optional<Scalar>& output_max);
};

class XNNPackTransposeConv2dOpContext final : public TransposeConv2dOpContext {
 private:
  ContextConv2D op_context_;
  // xnnpack convs use indirection buffer.
  // These buffers need setup at runtime and/or when input
  // dims change. If we are running the same model on multiple
  // threads, this can lead to contention where indirection buffer
  // is being accessed and updated at the same time from two different
  // threads.
  std::mutex xnnp_mutex_;

 public:
  XNNPackTransposeConv2dOpContext(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      const c10::optional<Scalar>& min,
      const c10::optional<Scalar>& max,
      ContextConv2D&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    output_padding_ = std::move(output_padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
    output_min_ = min;
    output_max_ = max;
    orig_weight_and_bias_freed_ = false;
  }

  Tensor run(const Tensor& input) override;
  void free_orig_weight_and_bias() override;

  static c10::intrusive_ptr<TransposeConv2dOpContext> create_context(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& output_padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      const c10::optional<Scalar>& output_min,
      const c10::optional<Scalar>& output_max);
};

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
