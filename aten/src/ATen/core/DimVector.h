#pragma once

#include <c10/util/SmallVector.h>
#include <stdint.h>

namespace at {

/// A container for sizes or strides
using DimVector = SmallVector<int64_t, 5>;

} // namespace at
