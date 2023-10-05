#pragma once

#include <c10/core/impl/LocalDispatchKeySet.h>

namespace at {
namespace impl {

// VmapMode contains a thread local count of how many nested vmaps
// we are currently inside. That number is known as the `vmap level`.
// VmapMode is used in the implementation of the Python `torch.vmap` API.
//
// NOTE: this is NOT the c++ api for torch.vmap. That doesn't exist yet.

struct TORCH_API VmapMode {
  // Returns the vmap level, aka the count of how many nested vmaps we're in.
  static int64_t current_vmap_level();

  // Increment the count of nested vmaps. If this causes the vmap level to be
  // greater than 0, then it enables DispatchKey::VmapMode on all tensors.
  static int64_t increment_nesting();

  // Decrements the count of nested vmaps. If this causes the vmap level to be
  // equal to 0, then it disables DispatchKey::VmapMode on all tensors.
  static int64_t decrement_nesting();
};

} // namespace impl
} // namespace at
