#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/irange.h>

namespace at { namespace native {

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
}}
