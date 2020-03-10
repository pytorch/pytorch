#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_sigmoid(const Tensor& self) {
  AT_ERROR("mkldnn_sigmoid: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  AT_ERROR("mkldnn_sigmoid_: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_sigmoid_backward(
    const Tensor& grad_output,
    const Tensor& output) {
  AT_ERROR("mkldnn_sigmoid_backward: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_sigmoid_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output) {
  AT_ERROR("mkldnn_sigmoid_backward_out: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_sigmoid(const Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_logistic_use_dst_for_bwd, ideep::prop_kind::forward);
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_logistic_use_dst_for_bwd, ideep::prop_kind::forward);
  return self;
}

Tensor mkldnn_sigmoid_backward(
    const Tensor& grad_output,
    const Tensor& output) {
  ideep::tensor& y = itensor_from_mkldnn(output);
  ideep::tensor& gy = itensor_from_mkldnn(grad_output);
  ideep::tensor gx;
  ideep::eltwise_backward::compute(y, gy, gx,
      ideep::algorithm::eltwise_logistic_use_dst_for_bwd);
  return new_with_itensor_mkldnn(std::move(gx), grad_output.options());
}

Tensor& mkldnn_sigmoid_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& output) {
  TORCH_CHECK(false, "mkldnn_sigmoid_backward_out: in-place mkldnn operation is not supported yet");
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
