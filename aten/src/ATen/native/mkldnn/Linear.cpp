#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  AT_ERROR("mkldnn_linear: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  AT_ERROR("mkldnn_linear_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  AT_ERROR("mkldnn_linear_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  AT_ERROR("mkldnn_linear_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  TORCH_CHECK(self.is_mkldnn(),
      "mkldnn_linear: input needs to be mkldnn layout");
  TORCH_CHECK(self.dim() >= 2,
      "mkldnn_linear: input needs to has dim at least 2, input dim ", self.dim());
  TORCH_CHECK(weight.dim() == 2,
      "mkldnn_linear: weight needs to be in 2 dim, weight dim", weight.dim());

  // reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
  auto self_reshaped = self.dim() > 2 ? self.reshape({-1, self.size(self.dim() - 1)}) : self;
  const ideep::tensor x = itensor_from_mkldnn(self_reshaped);
  const ideep::tensor w = itensor_from_tensor(weight).transpose_(0, 1);

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));
  
  ideep::tensor y;
  if (bias.defined()) {
    ideep::tensor b = itensor_from_tensor(bias);
    ideep::tensor::dims bias_dims(1, 1);
    bias_dims.push_back(output_size.back());
    b.reshape(bias_dims);
    ideep::matmul_forward::compute(x, w, b, y);
  } else {
    ideep::matmul_forward::compute(x, w, y);
  }

  if (self.dim() > 2) {
    return new_with_itensor_mkldnn(std::move(y), self.options()).reshape(output_size);
  }
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight){
  // grad_output can be [batch, M, N] or [M, N] 
  // if grad_output is [batch, M, N], we first reshape it to [batch*M, N]
  auto grad_output_reshaped = grad_output.dim() > 2 ?
      grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  const ideep::tensor w = itensor_from_tensor(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  ideep::tensor gradx;
  ideep::matmul_forward::compute(grady, w, gradx);

  if (input_size.size() > 2) {
    return new_with_itensor_mkldnn(std::move(gradx), grad_output.options()).reshape(input_size);
  }
  return new_with_itensor_mkldnn(std::move(gradx), grad_output.options());
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  // grad_output and input can be in two or three dims
  // if they are in three dims, first reshape them to two dims
  auto grad_output_reshaped = grad_output.dim() > 2 ?
      grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  const ideep::tensor grady = itensor_from_tensor(grad_output_reshaped).transpose_(0, 1);
  ideep::tensor& x = itensor_from_mkldnn(input_reshaped);
  ideep::tensor gradw;
  // for backward_weights, we currently fix the gradw as FP32 datatype, but in near future, it will support both FP32 weight for UX, and BF16 weight for best performance
  auto dst_type = get_mkldnn_dtype(weight.scalar_type());
  ideep::matmul_forward::compute(grady, x, gradw, 1.0f, 1.0f, ideep::scale_t(), 
      ideep::scale_t(), ideep::scale_t(), ideep::attr_t(), dst_type);
  
  Tensor gradb;
  if (bias_defined) {
    gradb = grad_output;
  }

  // Extract device info from weight and data type info from weight.
  // Since for current BF16 design, no matter the input is BF16 or FP32, the weight is FP32 tensor.
  if (weight.is_mkldnn()) {
    return std::tuple<Tensor, Tensor>{
        new_with_itensor_mkldnn(std::move(gradw), weight.options()), gradb};
  } else {
    return std::tuple<Tensor, Tensor>{
        mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw), weight.options())),
        mkldnn_to_dense(gradb, weight.scalar_type())};
  }
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
