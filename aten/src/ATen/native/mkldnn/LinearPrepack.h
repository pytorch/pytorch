#pragma once

#include <ATen/Tensor.h>
#include <ATen/native/mkldnn/Common.h>
#include <ATen/native/mkldnn/OpContext.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {
namespace internal {
namespace linear {

c10::intrusive_ptr<mkldnn::LinearOpContext> createLinearPrePackOpContext(
    Tensor weight,
    c10::optional<Tensor> bias,
    std::vector<int64_t> input_size,
    std::string attr,
    std::vector<c10::optional<at::Scalar>> scalars,
    c10::optional<std::string> algorithm);

Tensor linear_run(
    const Tensor& input,
    const c10::intrusive_ptr<mkldnn::LinearOpContext>& op_context);

ContextLinear create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef input_size,
    const ideep::attr_t& attr);

Tensor run(ContextLinear& context, const Tensor& input);

void run(ContextLinear& context, const Tensor& input, void* output);

} // namespace linear
} // namespace internal
} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()