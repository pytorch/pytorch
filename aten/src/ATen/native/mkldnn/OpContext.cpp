#include <ATen/native/mkldnn/ConvPrepack.h>
#include <ATen/native/mkldnn/OpContext.h>

#if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace mkldnn {

c10::intrusive_ptr<Conv2dOpContext> MkldnnConv2dOpContext::create_context(
    at::Tensor&& weight,
    c10::optional<at::Tensor>&& bias,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    std::vector<int64_t>&& input_size,
    const ideep::attr_t& attr) {
  auto op_context = mkldnn::internal::convolution2d::create(
      weight, bias, padding, stride, dilation, groups, input_size, attr);

  auto conv2d_op_context = c10::make_intrusive<MkldnnConv2dOpContext>(
      std::move(weight),
      std::move(bias),
      std::move(padding),
      std::move(stride),
      std::move(dilation),
      groups,
      std::move(input_size),
      std::move(op_context));

  return conv2d_op_context;
}

Tensor MkldnnConv2dOpContext::run(const Tensor& input) {
  return mkldnn::internal::convolution2d::run(op_context_, input);
}

} // namespace mkldnn
} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED()
