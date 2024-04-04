#pragma once

#include <ATen/Tensor.h>
#include <ATen/core/ivalue.h>
#include <ATen/native/mkldnn/Common.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {

const static std::map<std::string, ideep::attr_t> fusion_attr_map = {
    {"none", ideep::attr_t()},
    {"relu", ideep::attr_t::fuse_relu()},
};

using SerializationTypeConvPrePack = std::tuple<
    Tensor,
    c10::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    int64_t,
    std::vector<int64_t>,
    std::string>;

class ConvOpContext : public torch::jit::CustomClassHolder {
 protected:
  Tensor orig_weight_;
  c10::optional<Tensor> orig_bias_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  std::vector<int64_t> dilation_;
  int64_t groups_;
  std::vector<int64_t> input_size_;
  std::string attr_;

 public:
  SerializationTypeConvPrePack unpack() {
    return std::make_tuple(
        orig_weight_,
        orig_bias_,
        stride_,
        padding_,
        dilation_,
        groups_,
        input_size_,
        attr_);
  }

  virtual Tensor run(const Tensor& input) = 0;
  virtual void run(const Tensor& input, void* output) = 0;
};

class MkldnnConvOpContext final : public ConvOpContext {
 private:
  ContextConv op_context_;

 public:
  MkldnnConvOpContext(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      uint64_t groups,
      std::vector<int64_t>&& input_size,
      ContextConv&& op_context)
      : op_context_(std::move(op_context)) {
    orig_weight_ = std::move(weight);
    orig_bias_ = std::move(bias);
    padding_ = std::move(padding);
    stride_ = std::move(stride);
    dilation_ = std::move(dilation);
    groups_ = groups;
    input_size_ = std::move(input_size);
  }

  Tensor run(const Tensor& input) override;

  void run(const Tensor& input, void* output) override;

  static c10::intrusive_ptr<ConvOpContext> create_context(
      Tensor&& weight,
      c10::optional<Tensor>&& bias,
      std::vector<int64_t>&& padding,
      std::vector<int64_t>&& stride,
      std::vector<int64_t>&& dilation,
      int64_t groups,
      std::vector<int64_t>&& input_size,
      const ideep::attr_t& attr);
};

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
