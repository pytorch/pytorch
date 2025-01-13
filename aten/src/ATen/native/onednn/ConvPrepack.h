#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/onednn/Common.h>
#include <ATen/native/onednn/OpContext.h>

#if AT_ONEDNN_ENABLED()

namespace at::native::onednn::internal::convolution {

c10::intrusive_ptr<onednn::ConvOpContext> createConvPrePackOpContext(
    Tensor weight,
    std::optional<Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation,
    int64_t groups,
    std::vector<int64_t> input_size,
    std::string attr);

Tensor conv_run(
    const Tensor& input,
    const c10::intrusive_ptr<onednn::ConvOpContext>& op_context);

ContextConv create(
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const IntArrayRef input_size,
    const ideep::attr_t& attr);

Tensor run(ContextConv& context, const Tensor& input);

void run(ContextConv& context, const Tensor& input, void* output);

} // namespace at

#endif // AT_ONEDNN_ENABLED()
