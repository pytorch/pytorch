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

Tensor mkldnn_clone(const Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ERROR("mkldnn_clone: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("mkldnn_transpose: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("mkldnn_transpose_: ATen not compiled with MKLDNN support");
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
  if (self.sizes() == inferred_size) {
    return self;
  }
  const ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y{x};
  y.reshape<AllocForMKLDNN>({inferred_size.cbegin(), inferred_size.cend()});
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor mkldnn_clone(const Tensor& self, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  ideep::tensor& src = itensor_from_mkldnn(self);
  ideep::tensor dst;
  ideep::direct_copy::compute<AllocForMKLDNN>(src, dst);
  return new_with_itensor_mkldnn(std::move(dst), self.options());
}

Tensor mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  const ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  std::vector<int> axes(x.ndims());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[dim0], axes[dim1]);
  y.transpose_from<AllocForMKLDNN>(x, axes);
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("mkldnn_transpose_: in-place mkldnn operations are not supported yet");
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
