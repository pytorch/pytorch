#include <ATen/native/mkldnn/MKLDNNCommon.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor>
mkldnn_dropout(
    const Tensor& self,
    const double ratio) {
  TORCH_CHECK(false, "mkldnn_dropout: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_dropout_backward(
    const Tensor& grady,
    const Tensor& mask,
    double ratio) {
  TORCH_CHECK(false, "mkldnn_dropout_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

namespace at {
namespace native {

std::tuple<Tensor, Tensor>
mkldnn_dropout(
    const Tensor& self,
    double ratio) {
  TORCH_CHECK(
      ratio >= 0 && ratio < 1 && self.numel() != 0,
      "dropout probability has to be between 0 and 1, but got ",
      ratio);
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor mask;
  ideep::tensor y;
  ideep::dropout_forward::compute(x, ratio, y, mask);
  return std::tuple<Tensor, Tensor>{
      new_with_itensor_mkldnn(std::move(y), self.options()),
      new_with_itensor_mkldnn(std::move(mask), self.options())};
}

Tensor mkldnn_dropout_backward(
    const Tensor& grady,
    const Tensor& mask,
    double ratio) {
  if (ratio == 0 || grady.numel() == 0) {
    return grady;
  }
  ideep::tensor& dY = itensor_from_mkldnn(grady);
  ideep::tensor mask_mkldnn = itensor_from_mkldnn(mask);


  ideep::tensor dX;
  ideep::dropout_backward::compute(mask_mkldnn, dY, dX);
  return new_with_itensor_mkldnn(std::move(dX), grady.options());
}
} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
