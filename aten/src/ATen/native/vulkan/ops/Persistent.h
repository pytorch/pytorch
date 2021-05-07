#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

//
// This class is meant for allocation of resources that will persist through the
// execution of the program, or until they are explicitly free'd by this code's
// clients, and its usage pattern is in direct contrast with the primary resource
// pool from which tensors draw from.  Whereas the primary resource pool is
// purged in its entirety at the end of each inference run, the intended usage
// pattern for this class is such that it delegates object lifetime management
// to the users so resources can stick around for as long as required.  This
// is ideal for prepacked weights, or scnearios where a precomputed or
// once-transformed data can be stored and reused in subsequent runs.
//

struct Persistent final {
  api::Resource::Pool pool;
};

Persistent* persistent();

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
