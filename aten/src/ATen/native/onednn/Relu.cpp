#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/relu_native.h>                // for mkldnn_relu, mkldnn_...
#include <ATen/ops/threshold_backward_native.h>  // for mkldnn_relu_backward
#endif

#if !AT_ONEDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_relu_(Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu_: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_relu_backward(const Tensor& grad_output, const Tensor& input, const Scalar& threshold) {
  TORCH_CHECK(false, "mkldnn_relu_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>
#include <ATen/native/onednn/Utils.h>

namespace at::native {

Tensor mkldnn_relu(const Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(onednn_bf16_device_check(),
        "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  const ideep::tensor& x = itensor_from_onednn(input);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_onednn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor& mkldnn_relu_(Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(onednn_bf16_device_check(),
        "mkldnn_relu_: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  ideep::tensor& x = itensor_from_onednn(input);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return input;
}

Tensor mkldnn_relu_backward(const Tensor& grad_output, const Tensor& input, const Scalar& threshold) {
  ideep::tensor& x = itensor_from_onednn(input);
  ideep::tensor grady = itensor_from_onednn(grad_output);
  ideep::tensor gradx;
  ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_relu, /*alpha*/ 0.0);
  return new_with_itensor_onednn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

}

#endif // AT_ONEDNN_ENABLED
