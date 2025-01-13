#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/core/Tensor.h>
#include <c10/core/SymIntArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_mkldnn_reshape_native.h>
#include <ATen/ops/_mkldnn_transpose_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/view_native.h>
#endif

#if !AT_ONEDNN_ENABLED()

namespace at {
namespace native {

Tensor onednn_view(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with ONEDNN support");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with ONEDNN support");
}

Tensor onednn_clone(const Tensor& self, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "onednn_clone: ATen not compiled with ONEDNN support");
}

Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose: ATen not compiled with ONEDNN support");
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose_: ATen not compiled with ONEDNN support");
}

} // namespace native
} // namespace at

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>

namespace at::native {

Tensor onednn_view(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false,
      "Currently Onednn tensor does not support view. Change to use reshape instead");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }
  const ideep::tensor& x = itensor_from_onednn(self);
  ideep::tensor y{x};
  y.reshape(inferred_size);
  return new_with_itensor_onednn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor onednn_clone(const Tensor& self, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  ideep::tensor& src = itensor_from_onednn(self);
  ideep::tensor dst;
  ideep::direct_copy::compute(src, dst);
  return new_with_itensor_onednn(std::move(dst), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  const ideep::tensor& x = itensor_from_onednn(self);
  ideep::tensor y;
  std::vector<int> axes(x.ndims());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[dim0], axes[dim1]);
  y.transpose_from(x, axes);
  return new_with_itensor_onednn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose_: in-place onednn operations are not supported yet");
}

} // namespace at

#endif // AT_ONEDNN_ENABLED
