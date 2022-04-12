#pragma once

#include <c10/util/SmallVector.h>
#include <stdint.h>

namespace at {

constexpr size_t kDimVectorStaticSize = 5;

/// A container for sizes or strides
using DimVector = SmallVector<int64_t, kDimVectorStaticSize>;

} // namespace at
