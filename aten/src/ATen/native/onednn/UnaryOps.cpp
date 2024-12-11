#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sigmoid_native.h>          // for onednn_sigmoid, mkldnn_...
#include <ATen/ops/tanh_native.h>             // for onednn_tanh, onednn_tanh_
#endif

#if !AT_ONEDNN_ENABLED()

namespace at {
namespace native {

Tensor onednn_sigmoid(const Tensor& self) {
  TORCH_CHECK(false, "onednn_sigmoid: ATen not compiled with ONEDNN support");
}

Tensor& onednn_sigmoid_(Tensor& self) {
  TORCH_CHECK(false, "onednn_sigmoid_: ATen not compiled with ONEDNN support");
}

Tensor onednn_tanh(const Tensor& self) {
  TORCH_CHECK(false, "onednn_tanh: ATen not compiled with ONEDNN support");
}

Tensor& onednn_tanh_(Tensor& self) {
  TORCH_CHECK(false, "onednn_tanh_: ATen not compiled with ONEDNN support");
}

} // namespace native
} // namespace at

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>

namespace at::native {

Tensor onednn_sigmoid(const Tensor& self) {
  ideep::tensor& x = itensor_from_onednn(self);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return new_with_itensor_onednn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& onednn_sigmoid_(Tensor& self) {
  ideep::tensor& x = itensor_from_onednn(self);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return self;
}

Tensor onednn_tanh(const Tensor& self) {
  ideep::tensor& x = itensor_from_onednn(self);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_tanh, ideep::prop_kind::forward);
  return new_with_itensor_onednn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& onednn_tanh_(Tensor& self) {
  ideep::tensor& x = itensor_from_onednn(self);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_tanh, ideep::prop_kind::forward);
  return self;
}

} // namespace at

#endif // AT_ONEDNN_ENABLED
