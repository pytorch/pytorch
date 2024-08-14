#include <ATen/native/onednn/OpContext.h>

#if AT_ONEDNN_ENABLED()
#include <ATen/native/onednn/ConvPrepack.h>

namespace at {
namespace native {
namespace onednn {

c10::intrusive_ptr<ConvOpContext> OnednnConvOpContext::create_context(
    at::Tensor&& weight,
    std::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    std::vector<int64_t>&& input_size,
    const ideep::attr_t& attr) {
  auto op_context = onednn::internal::convolution::create(
      weight, bias, padding, stride, dilation, groups, input_size, attr);

  auto conv_op_context = c10::make_intrusive<OnednnConvOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      std::move(input_size),
      std::move(op_context));

  return conv_op_context;
}

Tensor OnednnConvOpContext::run(const Tensor& input) {
  return onednn::internal::convolution::run(op_context_, input);
}

void OnednnConvOpContext::run(const Tensor& input, void* output) {
  onednn::internal::convolution::run(op_context_, input, output);
}

} // namespace onednn
} // namespace native
} // namespace at

#endif // AT_ONEDNN_ENABLED()
