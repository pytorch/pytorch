#pragma once

#ifdef USE_XNNPACK

#include <ATen/Tensor.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace convolution2d {

class Conv2dPrePack final : public torch::OperatorKernel {
 public:
  c10::intrusive_ptr<xnnpack::XNNPackConv2dOpContext> operator()(
      Tensor weight,
      c10::optional<Tensor> bias,
      std::vector<int64_t> padding,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      int64_t groups);
};

class Conv2dPacked final : public torch::OperatorKernel {
 public:
  Tensor operator()(
      const Tensor& input,
      const c10::intrusive_ptr<xnnpack::XNNPackConv2dOpContext>& op_context);
};

ContextConv2D create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max);

} // namespace convolution2d
} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
