#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/copy_native.h>
#endif

#if !AT_ONEDNN_ENABLED()

namespace at {
namespace native {

Tensor& copy_onednn_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(false, "copy_onednn_: ATen not compiled with ONEDNN support");
}

} // namespace native
} // namespace at

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>

namespace at {
namespace native {

Tensor& copy_onednn_(Tensor& self, const Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      self.sizes() == src.sizes(),
      "copy_onednn_: only support same size tensor.");
  TORCH_CHECK(
      self.is_onednn() && src.is_onednn(),
      "copy_onednn_: between onednn layout and dense Tensors is not implemented! Found self type = ",
      self.toString(),
      " and src type = ",
      src.toString());
  ideep::tensor& x = itensor_from_onednn(src);
  ideep::tensor& y = itensor_from_onednn(self);
  ideep::direct_copy::compute(x, y);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_ONEDNN_ENABLED
