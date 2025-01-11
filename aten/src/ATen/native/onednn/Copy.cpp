#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/copy_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor& copy_mkldnn_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(false, "copy_mkldnn_: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/onednn/MKLDNNCommon.h>

namespace at::native {

Tensor& copy_mkldnn_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      self.sizes() == src.sizes(),
      "copy_mkldnn_: only support same size tensor.");
  TORCH_CHECK(
      self.is_mkldnn() && src.is_mkldnn(),
      "copy_mkldnn_: between mkldnn layout and dense Tensors is not implemented! Found self type = ",
      self.toString(),
      " and src type = ",
      src.toString());
  ideep::tensor& x = itensor_from_mkldnn(src);
  ideep::tensor& y = itensor_from_mkldnn(self);
  ideep::direct_copy::compute(x, y);
  return self;
}

} // namespace at

#endif // AT_MKLDNN_ENABLED
