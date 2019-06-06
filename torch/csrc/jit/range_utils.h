#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace torch {
namespace jit {
// Convert an python index (which may be negative) into an index usable for a
// C++ container
inline size_t normalizeIndex(int64_t idx, size_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

// Clamp `start` and `end` in the way Python does for iterable slicing.
inline std::pair<size_t, size_t> clamp_bounds(
    int64_t start,
    int64_t end,
    size_t list_size) {
  const size_t normalized_start =
      std::max((size_t)0, normalizeIndex(start, list_size));
  const size_t normalized_end =
      std::min(list_size, normalizeIndex(end, list_size));
  return std::pair<size_t, size_t>(normalized_start, normalized_end);
}
} // namespace jit
} // namespace torch