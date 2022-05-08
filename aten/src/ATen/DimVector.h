#pragma once
#include <c10/util/DimVector.h>

namespace at {

// Re-declaring 'DimVector' type and size inside 'at' namespace.
// This is done to avoid modifying every use into their 'c10'
// equivalent.

constexpr size_t kDimVectorStaticSize = c10::kDimVectorStaticSize;
using DimVector = c10::DimVector;

} // namespace at
