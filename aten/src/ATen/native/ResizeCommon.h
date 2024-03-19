#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

template <typename T>
inline T storage_size_for(ArrayRef<T> size, ArrayRef<T> stride) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(size.size() == stride.size(),
      "storage_size_for(size, stride) requires that size and stride ",
      "have the same size as a precondition.");
  T storage_size = 1;
  for (const auto dim : c10::irange(size.size())) {
    if (size[dim] == 0) {
      storage_size = 0;
      break;
    }
    storage_size += (size[dim] - 1) * stride[dim];
  }
  return storage_size;
}

inline const Tensor& resize_named_tensor_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(self.has_names());
  TORCH_CHECK(
      self.sizes() == size,
      "Cannot resize named tensor with resize_ or resize_as_ (tried to resize "
      "Tensor",
      self.names(),
      " with size ",
      self.sizes(),
      " to ",
      size,
      "). This may be caused by passing a named tensor ",
      "as an `out=` argument; please ensure that the sizes are the same. ");
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "Unsupported memory format for named tensor resize ",
      optional_memory_format.value());
  return self;
}

// For deterministic output, fill new elements that were added after a storage
// resize with NaN or MAX_INT. `old_storage_nbytes` is the size of the storage
// before the resize happened.
inline const Tensor& fill_resize_deterministic_(const Tensor& tensor, int64_t old_storage_nbytes) {
  const at::Storage& storage = tensor.unsafeGetTensorImpl()->unsafe_storage();
  int64_t new_storage_nbytes = storage.nbytes();
  int64_t old_storage_numel = old_storage_nbytes / tensor.itemsize();
  int64_t new_storage_numel = new_storage_nbytes / tensor.itemsize();
  if (new_storage_numel > old_storage_numel) {
    at::Tensor tensor_view = at::empty({}, at::TensorOptions().dtype(tensor.scalar_type()).device(tensor.device()));
    tensor_view.set_(
      storage,
      /*storage_offset=*/old_storage_numel,
      /*size=*/{new_storage_numel - old_storage_numel},
      /*stride=*/{1});
    at::native::fill_empty_deterministic_(tensor_view);
  }
  return tensor;
}

} // namespace at::native
