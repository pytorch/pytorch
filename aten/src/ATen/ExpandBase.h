#include <ATen/core/TensorBase.h>

// Broadcasting utilities for working with TensorBase
namespace at {
namespace internal {
TORCH_API TensorBase expand_slow_path(const TensorBase &self, IntArrayRef size);
} // namespace internal

inline c10::MaybeOwned<TensorBase> expand_size(const TensorBase &self, IntArrayRef size) {
  if (size.equals(self.sizes())) {
    return c10::MaybeOwned<TensorBase>::borrowed(self);
  }
  return c10::MaybeOwned<TensorBase>::owned(
      at::internal::expand_slow_path(self, size));
}
c10::MaybeOwned<TensorBase> expand_size(TensorBase &&self, IntArrayRef size) = delete;

inline c10::MaybeOwned<TensorBase> expand_inplace(const TensorBase &tensor, const TensorBase &to_expand) {
  return expand_size(to_expand, tensor.sizes());
}
c10::MaybeOwned<TensorBase> expand_inplace(const TensorBase &tensor, TensorBase &&to_expand) = delete;

} // namespace at
