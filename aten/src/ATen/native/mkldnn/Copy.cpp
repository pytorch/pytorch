#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/mkldnn/Copy.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& mkldnn_copy_from(Tensor& self, const Tensor& src) {
  AT_ERROR("mkldnn_copy_from: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

Tensor& mkldnn_copy_from(Tensor& self, const Tensor& src) {
  TORCH_CHECK(
      src.scalar_type() == at::kFloat || src.scalar_type() == at::kBFloat16,
      "Mkldnn copy only works with kFloat and kBfloat16 as source Tensor");
  auto &stensor = itensor_from_mkldnn(src);
  auto &dtensor = itensor_from_mkldnn(self);
  dtensor.feed_from(stensor);
  return self;
}

} // namespace native
} // namespace at
#endif // AT_MKLDNN_ENABLED
