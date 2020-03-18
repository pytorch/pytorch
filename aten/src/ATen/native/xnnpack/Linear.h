#pragma once

#ifdef USE_XNNPACK

#include <ATen/Tensor.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace at {
namespace native {
namespace xnnpack {
namespace internal {
namespace linear {

class LinearPrePack final : public torch::OperatorKernel {
  public:
  c10::intrusive_ptr<xnnpack::XNNPackLinearOpContext> operator()(
      Tensor weight,
      c10::optional<Tensor> bias);
};

class LinearPacked final : public torch::OperatorKernel {
  public:
  Tensor operator()(
      const Tensor& input,
      const c10::intrusive_ptr<xnnpack::XNNPackLinearOpContext>& op_context);
};

ContextLinear create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const float output_min,
    const float output_max);
} // namespace linear
} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
