#pragma once

#ifdef USE_XNNPACK

#include <ATen/Tensor.h>
#include <ATen/native/xnnpack/Common.h>
#include <ATen/native/xnnpack/OpContext.h>

namespace at::native::xnnpack {
namespace internal::convolution2d {

c10::intrusive_ptr<xnnpack::Conv2dOpContext>
    createConv2dClampPrePackOpContext(
        Tensor weight,
        c10::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        const c10::optional<Scalar>& output_min,
        const c10::optional<Scalar>& output_max);

c10::intrusive_ptr<xnnpack::TransposeConv2dOpContext>
    createConv2dTransposeClampPrePackOpContext(
        Tensor weight,
        c10::optional<Tensor> bias,
        std::vector<int64_t> stride,
        std::vector<int64_t> padding,
        std::vector<int64_t> output_padding,
        std::vector<int64_t> dilation,
        int64_t groups,
        const c10::optional<Scalar>& output_min,
        const c10::optional<Scalar>& output_max);

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::Conv2dOpContext>& op_context);

IValue
unpack_prepacked_sizes_conv2d(const IValue& ivalue);

Tensor conv2d_transpose_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<xnnpack::TransposeConv2dOpContext>& op_context);

ContextConv2D create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef output_padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const bool transposed,
    const float output_min,
    const float output_max);

Tensor run(ContextConv2D& context, const Tensor& input);

} // namespace internal::convolution2d

Tensor convolution2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups);
} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */
