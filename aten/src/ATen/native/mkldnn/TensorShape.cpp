#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  AT_ERROR("mkldnn_reshape: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  AT_ERROR("mkldnn_reshape: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  AT_ERROR(
      "Currently Mkldnn tensor does not support view. Change to use reshape instead");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  const ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  ideep::direct_copy::compute<AllocForMKLDNN>(x, y);
  y.reshape({inferred_size.cbegin(), inferred_size.cend()});
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
