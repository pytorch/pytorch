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

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  const ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_mkldnn(std::move(y), input.options());
}

Tensor& mkldnn_relu_(Tensor& input) {
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return input;
}

}}

#endif // AT_MKLDNN_EBABLED
