#include <Linear.h>
#include <FusionUtils.h>
#include <torch/library.h>


namespace at {
namespace native::xpu {

using namespace impl;
#define IPEX_LINEAR_DEFINATION(func)                                       \
  Tensor linear_##func(                                                    \
      const Tensor& input,                                                 \
      const Tensor& weight,                                                \
      const c10::optional<Tensor>& bias) {                                 \
    RECORD_FUNCTION(                                                       \
        "linear_" #func, std::vector<c10::IValue>({input, weight, bias})); \
    auto linear_wrapper = LinearConverter();                               \
    auto post_op = [=]() {                                                 \
      Attr attr;                                                           \
      attr.append_post_eltwise(                                            \
          /* scale */ 1.f,                                                 \
          /* alpha */ 0.f,                                                 \
          /* beta */ 0.f,                                                  \
          attr.kind_with_##func);                                          \
      return attr;                                                         \
    };                                                                     \
    Tensor result;                                                         \
    return linear_wrapper.call(result, input, weight, bias, post_op);      \
  }

#define IPEX_LINEAR_BINARY_DEFINATION(func)                                \
  Tensor linear_binary_##func(                                             \
      const Tensor& input,                                                 \
      const Tensor& weight,                                                \
      const c10::optional<Tensor>& bias,                                   \
      const Tensor& binary) {                                              \
    RECORD_FUNCTION(                                                       \
        "linear_binary_" #func,                                            \
        std::vector<c10::IValue>({input, weight, bias}));                  \
    auto linear_wrapper = LinearConverter();                               \
    auto post_op = [=]() {                                                 \
      Attr attr;                                                           \
      attr.append_scale_binary(attr.kind_with_binary_##func, binary, 1.f); \
      return attr;                                                         \
    };                                                                     \
    Tensor result;                                                         \
    result = linear_wrapper.call(result, input, weight, bias, post_op);    \
    if (!linear_wrapper.is_fused()) {                                      \
      result = at::func(result, binary);                                   \
    }                                                                      \
    return result;                                                         \
  }

Tensor linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    c10::string_view attr,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm) {
  xpu::onednn::Attr att;
  att = construct_unary_attr(attr, scalars, algorithm, att);
  auto linear_wrapper = LinearConverter();

  Tensor result;
  return linear_wrapper.call(result, input_t, weight_t, bias_opt, att);
}

Tensor linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    c10::string_view binary_attr) {
  Tensor output;

  xpu::onednn::Attr attr;
  attr = construct_binary_attr(binary_attr, /*alpha*/ 1.f, other_t, attr);

  Tensor _input =
      input_t.dim() <= 2 ? input_t : input_t.contiguous();
  auto linear_wrapper = LinearConverter();
  Tensor result;
  return linear_wrapper.call(result, input_t, weight_t, bias_opt, attr);
}

TORCH_LIBRARY_IMPL(mkldnn, XPU, m){
  m.impl(
    TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise"),
    TORCH_FN(linear_pointwise)
  );
  m.impl(
    TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise.binary"),
    TORCH_FN(linear_pointwise_binary)
  );
}

} // namespace native::xpu
} // namespace at
