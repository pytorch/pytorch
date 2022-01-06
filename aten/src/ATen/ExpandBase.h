#include <ATen/core/TensorBase.h>

// Broadcasting utilities for working with TensorBase
namespace at {
namespace internal {
TensorBase expand_slow_path(const TensorBase &self, IntArrayRef size);
}

inline c10::MaybeOwned<TensorBase> expand_to_size(const TensorBase &self, IntArrayRef size) {
  if (size.equals(self.sizes())) {
    return c10::MaybeOwned<TensorBase>::borrowed(self);
  }
  return c10::MaybeOwned<TensorBase>::owned(
      at::internal::expand_slow_path(self, size));
}

inline c10::MaybeOwned<TensorBase> expand_inplace(const TensorBase &tensor, const TensorBase &to_expand) {
  return expand_to_size(to_expand, tensor.sizes());
}

}
