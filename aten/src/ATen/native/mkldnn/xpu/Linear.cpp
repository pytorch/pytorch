#include <Linear.h>
#include <FusionUtils.h>
#include <torch/library.h>


namespace at {
namespace native::xpu {

Tensor linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const c10::optional<Tensor>& bias_opt,
    c10::string_view attr,
    torch::List<c10::optional<at::Scalar>> scalars,
    c10::optional<c10::string_view> algorithm) {
  onednn::Attr att;
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

  onednn::Attr attr;
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
