#pragma once

#include <ATen/core/Tensor.h>

namespace at::native {

// Packed float shell dtypes (see c10::isPackedShellFloatType) have no typed
// kernels. Ops that only move or compare whole elements run on this uint8 view,
// which appends a trailing dim of the element size and aliases t's storage.
inline Tensor as_element_bytes(const Tensor& t) {
  return t.unsqueeze(-1).view(kByte);
}

// Inverse of as_element_bytes. bytes must have a trailing dim of
// elementSize(dtype) bytes with stride 1.
inline Tensor from_element_bytes(const Tensor& bytes, ScalarType dtype) {
  return bytes.view(dtype).squeeze(-1);
}

// Index for gather/scatter on as_element_bytes views: repeats every index
// across the trailing element-byte dim.
inline Tensor expand_index_to_element_bytes(
    const Tensor& index,
    int64_t element_size) {
  auto sizes = index.sizes().vec();
  sizes.push_back(element_size);
  return index.unsqueeze(-1).expand(sizes);
}

} // namespace at::native
