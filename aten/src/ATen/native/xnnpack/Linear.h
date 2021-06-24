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

c10::intrusive_ptr<xnnpack::LinearOpContext> createLinearClampPrePackOpContext(
    Tensor weight,
    c10::optional<Tensor> bias,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max);

Tensor linear_clamp_run(const Tensor& input, const c10::intrusive_ptr<xnnpack::LinearOpContext>& op_context);

ContextLinear create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const float output_min,
    const float output_max);

Tensor run(const ContextLinear& context, const Tensor& input);
} // namespace linear
} // namespace internal
} // namespace xnnpack
} // namespace native
} // namespace at

#endif /* USE_XNNPACK */
