#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_relu_(Tensor& input) {
  TORCH_CHECK(false, "mkldnn_relu_: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  const ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt());
}

Tensor& mkldnn_relu_(Tensor& input) {
  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_relu_: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return input;
}

}}

#endif // AT_MKLDNN_EBABLED
