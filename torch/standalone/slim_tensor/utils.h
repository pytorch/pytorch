#pragma once
#include <cstdint>
#include <stdexcept>

// OK to use c10 headers here because their corresponding cpp files will be
// included in the final binary.
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <torch/standalone/slim_tensor/array_ref.h>

namespace torch::standalone {
inline size_t compute_numel(const ArrayRef& sizes) {
  int64_t numel = 1;
  for (auto& s : sizes) {
    numel *= s;
  }
  return numel;
}

inline size_t compute_nbytes(const ArrayRef& sizes, c10::ScalarType dtype) {
  return compute_numel(sizes) * c10::elementSize(dtype);
}

inline size_t compute_nbytes(size_t numel, c10::ScalarType dtype) {
  return numel * c10::elementSize(dtype);
}
} // namespace torch::standalone
