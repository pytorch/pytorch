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
  typedef api::Handle<
      api::Resource::Buffer,
      std::function<void(const api::Resource::Buffer&)>> Buffer;

  typedef api::Handle<
      api::Resource::Image,
      std::function<void(const api::Resource::Image&)>> Image;

  class Pool final {
   public:
    explicit Pool(const api::GPU& gpu);
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = default;
    Pool& operator=(Pool&&) = default;
    ~Pool() = default;

    Buffer buffer(
        const IntArrayRef sizes,
        const TensorOptions& options);

    Image image(
        const VkExtent3D& extents,
        const TensorOptions& options);

   private:
    api::Resource::Pool pool_;
  };
};

Persistent::Pool* persistent();

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
