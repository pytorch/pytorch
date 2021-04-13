#pragma once

#include "lazy_tensors/span.h"

namespace lazy_tensors {

inline lazy_tensors::Span<const int64> AsInt64Slice(
    lazy_tensors::Span<const int64> slice) {
  return slice;
}

}  // namespace lazy_tensors
