#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/native/Activation.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/gelu_backward_native.h>
#endif

#if !AT_ONEDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_gelu(const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "mkldnn_gelu: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(false, "mkldnn_gelu_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>
#include <ATen/native/onednn/Utils.h>

namespace at { namespace native {

Tensor mkldnn_gelu(const Tensor& input, c10::string_view approximate) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_gelu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "mkldnn_gelu: fast, approximate gelu is not supported");
  const ideep::tensor& x = itensor_from_tensor(input);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_gelu_erf, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, c10::string_view approximate) {
  TORCH_CHECK(get_gelutype_enum(approximate) == GeluType::None,
                  "mkldnn_gelu_backward: fast, approximate gelu is not supported");
  const ideep::tensor& x = itensor_from_tensor(input);
  ideep::tensor grady = itensor_from_tensor(grad_output);
  ideep::tensor gradx;
  ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_gelu_erf, /*alpha*/ 0.0);
  return new_with_itensor_mkldnn(std::move(gradx),
                                 optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

}}

#endif // AT_ONEDNN_ENABLED
