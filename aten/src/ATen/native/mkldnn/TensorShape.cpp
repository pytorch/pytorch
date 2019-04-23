#include <ATen/Config.h>
#include <ATen/ATen.h>
#include <numeric>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_clone(const Tensor& self) {
  AT_ERROR("mkldnn_clone: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor mkldnn_clone(const Tensor& self) {
  ideep::tensor& src = itensor_from_mkldnn(self);
  ideep::tensor& dst{src};
  return new_with_itensor_mkldnn(std::move(dst), self.options());
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
