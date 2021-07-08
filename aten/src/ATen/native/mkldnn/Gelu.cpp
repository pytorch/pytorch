#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_gelu(const Tensor& input) {
  TORCH_CHECK(false, "mkldnn_gelu: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input) {
  TORCH_CHECK(false, "mkldnn_gelu_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/mkldnn/Gelu.h>

namespace at { namespace native {

Tensor _mkldnn_gelu(const Tensor& input, const Tensor& result) {
  const ideep::tensor& x = itensor_from_tensor(input);
  ideep::tensor y;
  if (!input.is_mkldnn()) {
    // deal with dense layout
    result.resize_(input.sizes(), input.suggest_memory_format());
    y = itensor_from_tensor(result);
  }
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_gelu_erf, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  if (!input.is_mkldnn()) {
    return result;
  }
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor mkldnn_gelu(const Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_gelu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  Tensor result = at::empty({0}, input.options());
  return _mkldnn_gelu(input, result);
}

Tensor _mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& grad_input) {
  const ideep::tensor& x = itensor_from_tensor(input);
  ideep::tensor grady = itensor_from_tensor(grad_output);
  ideep::tensor gradx;
  if (!input.is_mkldnn()) {
    // deal with dense layout
    grad_input.resize_(grad_output.sizes(), grad_output.suggest_memory_format());
    gradx = itensor_from_tensor(grad_input);
  }
  ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_gelu_erf, /*alpha*/ 0.0);
  if (!input.is_mkldnn()) {
    return grad_input;
  }
  return new_with_itensor_mkldnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input) {
  Tensor grad_input = at::empty({0}, grad_output.options());
  return _mkldnn_gelu_backward(grad_output, input, grad_input);
}

}}

#endif // AT_MKLDNN_EBABLED
