#include <FusionUtils.h>
#include <Linear.h>
#include <torch/library.h>

namespace at::native::xpu {

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
    
    onednn::Attr attr;
    attr = construct_binary_attr<true>(binary_attr, /*alpha*/1.f, other_t, attr);
    auto input = input_t.contiguous();

    auto input_size = input.sizes();
    const int64_t dim = input.dim();

    // dim collapse
    auto input_reshaped = dim == 2 ? input : input.reshape({-1, input.size(input.dim() -1)});
    std::vector<int64_t> output_size(input_size.begin(), input_size.end()-1);
    output_size.push_back(weight_t.size(0));
    
    auto output = at::empty(output_size, input.options());
    auto other_reshaped = other_t.contiguous();
    if(dim != 2){
      // input [m, k], weight [n, k], output [m, n]
      std::vector<int64_t> output_size_reshaped = {
        input_reshaped.size(0), weight_t.size(0)
      };
      other_reshaped = other_reshaped.reshape(output_size_reshaped);
    }else{
      TORCH_CHECK(output.dim() == other_reshaped.dim(),
        "linear_binary_run expects the dimension of output and other tensor to be the same");
    }

    auto bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
    at::native::onednn::matmul(
        other_reshaped, input_reshaped, weight_t, bias, /*m2_trans*/ false, attr);

    if(dim != 2){
      output = output.reshape(output_size);
    }
    return output;

}

Tensor linear_pointwise_binary_(
  const Tensor& input_t,
  const Tensor& other_t,
  const Tensor& weight_t,
  const c10::optional<Tensor>& bias_opt,
  c10::string_view binary_attr) {
    onednn::Attr attr;
    attr = construct_binary_attr<true>("sum", /*alpha*/1.f, other_t, attr);
    auto input = input_t.contiguous();

    auto input_size = input.sizes();
    const int64_t dim = input.dim();

    // dim collapse
    auto input_reshaped = dim == 2 ? input : input.reshape({-1, input.size(input.dim() -1)});
    std::vector<int64_t> output_size(input_size.begin(), input_size.end()-1);
    output_size.push_back(weight_t.size(0));
    
    auto output = at::empty(output_size, input.options());
    auto other_reshaped = other_t.contiguous();
    if(dim != 2){
      // input [m, k], weight [n, k], output [m, n]
      std::vector<int64_t> output_size_reshaped = {
        input_reshaped.size(0), weight_t.size(0)
      };
      other_reshaped = other_reshaped.reshape(output_size_reshaped);
    }else{
      TORCH_CHECK(output.dim() == other_reshaped.dim(),
        "linear_binary_run expects the dimension of output and other tensor to be the same");
    }

    auto bias = bias_opt.has_value() ? bias_opt.value() : at::Tensor();
    at::native::onednn::matmul(
        other_reshaped, input_reshaped, weight_t, bias, /*m2_trans*/ false, attr);

    if(dim != 2){
      output = output.reshape(output_size);
    }
    return output;

  }


TORCH_LIBRARY_IMPL(mkldnn, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise"),
      TORCH_FN(linear_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise.binary"),
      TORCH_FN(linear_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise_.binary"),
      TORCH_FN(linear_pointwise_binary_));
}

} // namespace at::native::xpu
