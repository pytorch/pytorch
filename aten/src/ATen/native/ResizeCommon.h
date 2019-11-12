#pragma once

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>

namespace at { namespace native {

#ifdef BUILD_NAMEDTENSOR
inline Tensor& resize_named_tensor_(
    Tensor& self,
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
      optional_memory_format.value_or(MemoryFormat::Contiguous) ==
          MemoryFormat::Contiguous,
      "Unsupported memory format for named tensor resize ",
      optional_memory_format.value());
  return self;
}
#endif
}}
