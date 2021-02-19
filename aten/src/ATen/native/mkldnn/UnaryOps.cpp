#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_sigmoid(const Tensor& self) {
  TORCH_CHECK(false, "mkldnn_sigmoid: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  TORCH_CHECK(false, "mkldnn_sigmoid_: ATen not compiled with MKLDNN support");
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
      x, y, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
