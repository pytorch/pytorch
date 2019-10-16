#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_relu(const Tensor& input) {
  AT_ERROR("mkldnn_relu: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_relu_(Tensor& input) {
  AT_ERROR("mkldnn_relu_: ATen not compiled with MKLDNN support");
}

Tensor dnnl_relu(const Tensor& input) {
  AT_ERROR("dnnl_relu: ATen not compiled with MKLDNN support");
}

Tensor& dnnl_relu_(Tensor& input) {
  AT_ERROR("dnnl_relu_: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at { namespace native {

//
// These two new interface have extended semantics,
// if input is an Opaque tensor supported by this op, it will return an Opaque tensor.
// if input is CPUTensor, it returns CPUTensor
//
Tensor dnnl_relu(const Tensor& input) {
  const ideep::tensor x = get_mkldnn_tensor(input);

  // Create CPU or Opaque result tensor
  auto output = dnnl_empty_like(input);
  auto y = get_mkldnn_tensor(output);

  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_relu,
      ideep::prop_kind::forward_inference, /*alpha*/ 0.0);
  return output;
}

Tensor& dnnl_relu_(Tensor& input) {
  ideep::tensor x = get_mkldnn_tensor(input);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_relu,
      ideep::prop_kind::forward_inference, /*alpha*/ 0.0);
  return input;
}

Tensor mkldnn_relu(const Tensor& input) {
  const ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor y;
  ideep::eltwise_forward::compute<AllocForMKLDNN>(
      x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_mkldnn(std::move(y), input.options());
}

Tensor& mkldnn_relu_(Tensor& input) {
  return native::dnnl_relu_(input);
}

}}

#endif // AT_MKLDNN_EBABLED
