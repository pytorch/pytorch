#pragma once

#ifdef USE_XNNPACK

#include <ATen/Tensor.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace at::native::xnnpack {
namespace internal::linear {

c10::intrusive_ptr<xnnpack::LinearOpContext> createLinearClampPrePackOpContext(
    Tensor weight,
    std::optional<Tensor> bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max);

Tensor linear_clamp_run(const Tensor& input, const c10::intrusive_ptr<xnnpack::LinearOpContext>& op_context);

IValue
unpack_prepacked_sizes_linear(const IValue& ivalue);

ContextLinear create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const float output_min,
    const float output_max);

Tensor run(const ContextLinear& context, const Tensor& input);
} // namespace internal::linear

bool use_linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias);

Tensor linear(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias);

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
