#pragma once

#ifdef USE_XNNPACK

#include <ATen/core/ivalue.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/Tensor.h>

namespace at {
namespace native {
namespace xnnpack {

using SerializationTypeLinearPrePack = std::tuple<Tensor, c10::optional<Tensor>>;
using SerializationTypeConv2dPrePack =
  std::tuple<Tensor, c10::optional<Tensor>,
  std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, int64_t>;

class XNNPackLinearOpContext : public torch::jit::CustomClassHolder {
  private:
    Tensor orig_weight_;
    c10::optional<Tensor> orig_bias_;
    ContextLinear op_context_;

  public:
    XNNPackLinearOpContext(Tensor&& weight,
        c10::optional<Tensor>&& bias) {
      orig_weight_ = std::move(weight);
      orig_bias_ = std::move(bias);
    }
    //XNNPackLinearOpContext(const XNNPackLinearOpContext&) = delete; // Copy constructor
    //XNNPackLinearContext& operator=(const XNNPackLinearOpContext&) = delete; // Assign
    //Move construct and move assign?
    const ContextLinear& get_context() const {
      return op_context_;
    }

    SerializationTypeLinearPrePack unpack() {
      return std::make_tuple(orig_weight_, orig_bias_);
    }

    static c10::intrusive_ptr<XNNPackLinearOpContext> create_context(Tensor&& weight,
        c10::optional<Tensor>&& bias,
        const c10::optional<double> output_min,
        const c10::optional<double> output_max);
};

class XNNPackConv2dOpContext : public torch::jit::CustomClassHolder {
  private:
    Tensor orig_weight_;
    c10::optional<Tensor> orig_bias_;
    std::vector<int64_t> padding_;
    std::vector<int64_t> stride_;
    std::vector<int64_t> dilation_;
    int64_t groups_;
    ContextConv2D op_context_;

  public:
    XNNPackConv2dOpContext(Tensor&& weight,
        c10::optional<Tensor>&& bias,
        std::vector<int64_t>&& padding,
        std::vector<int64_t>&& stride,
        std::vector<int64_t>&& dilation,
        uint64_t groups
        ) {
      orig_weight_ = std::move(weight);
      orig_bias_ = std::move(bias);
      padding_ = std::move(padding);
      stride_ = std::move(stride);
      dilation_ = std::move(dilation);
      groups_ = groups;
    }
    //XNNPackLinearOpContext(const XNNPackLinearOpContext&) = delete; // Copy constructor
    //XNNPackLinearContext& operator=(const XNNPackLinearOpContext&) = delete; // Assign
    //Need to define Move construct and move assign?
    const ContextConv2D& get_context() const {
      return op_context_;
    }

    SerializationTypeConv2dPrePack unpack() {
      return std::make_tuple(orig_weight_, orig_bias_, padding_,
          stride_, dilation_, groups_);
    }

    static c10::intrusive_ptr<XNNPackConv2dOpContext> create_context(Tensor&& weight,
        c10::optional<Tensor>&& bias,
        std::vector<int64_t>&& padding,
        std::vector<int64_t>&& stride,
        std::vector<int64_t>&& dilation,
        int64_t groups,
        const c10::optional<double> output_min,
        const c10::optional<double> output_max);
};
} // xnnpack

} // native
} // at

#endif /* USE_XNNPACK */
