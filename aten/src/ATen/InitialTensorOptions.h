#pragma once

#include <c10/core/TensorOptions.h>

namespace at {

// Represents the initial TensorOptions, before the "defaults" are ever changed.
// This is designed to be used in library code, where the explicit devices, dtypes, etc. are known.
// NOTE: this is not a stable API.
inline TensorOptions initialTensorOptions() {
  return TensorOptions(kCPU).dtype(kFloat).layout(kStrided)
                            .requires_grad(false);
}

}
