#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight, const c10::optional<Tensor>& bias_opt) {
  TORCH_CHECK(false, "mkldnn_linear: ATen not compiled with MKLDNN support");
}
Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_linear_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_linear_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_linear_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight_t, const c10::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});

  const int64_t dim = self.dim();
  TORCH_CHECK(
      self.dim() != 0,
      "mkldnn_linear: input needs to has dim at least 1, input dim ",
      self.dim());
  TORCH_CHECK(self.is_mkldnn(),
      "mkldnn_linear: input needs to be mkldnn layout");
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_linear: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  // reshape first if input dim != 2 and the reshape will cost a memory copy.
  auto self_reshaped =
      dim == 2 ? self : self.reshape({-1, self.size(self.dim() - 1)});

  const ideep::tensor x = itensor_from_mkldnn(self_reshaped);
  // weight_t can be a mkldnn tensor or dense tensor.
  const Tensor weight = (weight_t.is_mkldnn() || weight_t.is_contiguous()) ? weight_t : weight_t.contiguous();
  const ideep::tensor w = itensor_from_tensor(weight);

  ideep::tensor y;
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() != 2) {
    return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                   self.options().device_opt()).reshape(output_size);
  }
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}


Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight_t){
  TORCH_CHECK(grad_output.is_mkldnn(),
      "mkldnn_linear_backward: grad_output needs to be mkldnn layout");
  TORCH_CHECK(weight_t.device().is_cpu() && weight_t.scalar_type() == kFloat,
      "mkldnn_linear_backward: weight_t needs to be a dense tensor");
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;

  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  // weight_t always dense tensor for training.
  const Tensor weight = weight_t.is_contiguous() ? weight_t : weight_t.contiguous();
  const ideep::tensor w = itensor_view_from_dense(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  ideep::tensor gradx;
  ideep::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);

  if (input_size.size() > 2) {
    return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                   grad_output.options().device_opt()).reshape(input_size);
  }
  return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(grad_output.is_mkldnn() && input.is_mkldnn(),
      "mkldnn_linear_backward: grad_output and input needs to be mkldnn layout");
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == kFloat,
      "mkldnn_linear_backward: weight needs to be a dense tensor");

  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  ideep::tensor& x = itensor_from_mkldnn(input_reshaped);
  ideep::tensor gradw, gradb;
  if (bias_defined) {
    ideep::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    ideep::inner_product_backward_weights::compute(x, grady, gradw);
  }

  return std::tuple<Tensor, Tensor>{
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt())),
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt()))};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, std::array<bool,3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
