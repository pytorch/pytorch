#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  AT_ERROR("mkldnn_add_out: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  AT_ERROR("mkldnn_add: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  AT_ERROR("mkldnn_add_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor& mkldnn_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor& z = itensor_from_mkldnn(result);
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute<AllocForMKLDNN>(scales, {x, y}, z);

  return result;
}

Tensor mkldnn_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor& y = itensor_from_mkldnn(other);

  ideep::tensor z;
  const std::vector<float> scales{1.0, alpha.to<float>()};
  ideep::sum::compute<AllocForMKLDNN>(scales, {x, y}, z);

  return new_with_itensor_mkldnn(std::move(z), self.options());
}

Tensor& mkldnn_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::mkldnn_add_out(self, self, other, alpha);
}


} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
