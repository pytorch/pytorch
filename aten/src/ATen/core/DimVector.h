#pragma once
#include <c10/util/DimVector.h>

namespace at {

// Re-declaring 'DimVector' type and size inside 'at' namespace.
// This is done to avoid modifying every use into their 'c10'
// equivalent.

using c10::DimVector;
using c10::kDimVectorStaticSize;

} // namespace at
