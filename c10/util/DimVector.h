#pragma once

#include <c10/core/impl/SizesAndStrides.h>
#include <c10/util/SmallVector.h>
#include <cstdint>

namespace c10 {

constexpr size_t kDimVectorStaticSize = C10_SIZES_AND_STRIDES_MAX_INLINE_SIZE;

/// A container for sizes or strides
using DimVector = SmallVector<int64_t, kDimVectorStaticSize>;

} // namespace c10
