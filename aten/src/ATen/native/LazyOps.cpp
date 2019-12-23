#include <ATen/core/LazyTensor.h>

namespace at {
namespace native {

CAFFE2_API bool is_lazy(at::Tensor const& self) {
  return self.unsafeGetTensorImpl()->is_lazy();
}

CAFFE2_API at::Tensor to_lazy(at::Tensor const& self) {
  return detail::make_tensor<LazyTensorImpl>(self);
}

CAFFE2_API at::Tensor to_eager(at::Tensor const& self) {
  TORCH_CHECK(is_lazy(self));
  auto lt = static_cast<LazyTensorImpl*>(self.unsafeGetTensorImpl());
  auto tensor = lt->to_eager();
  return tensor;
}

} // namespace native
} // namespace at
