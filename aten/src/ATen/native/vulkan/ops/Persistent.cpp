#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

void release_buffer(
    std::reference_wrapper<api::Resource::Pool> pool,
    const api::Resource::Buffer& buffer) {
  pool.get().release(buffer);
}

void release_image(
    std::reference_wrapper<api::Resource::Pool> pool,
    const api::Resource::Image& image) {
  pool.get().release(image);
}

} // namespace

Persistent::Pool::Pool(const api::GPU& gpu)
  : pool_(gpu) {
}

Persistent::Buffer Persistent::Pool::buffer(
    const IntArrayRef sizes,
    const TensorOptions& options) {
  return {
    // Resource
    pool_.buffer({
        // Size
        std::accumulate(
            sizes.cbegin(),
            sizes.cend(),
            c10::elementSize(c10::typeMetaToScalarType(options.dtype())),
            std::multiplies<int64_t>()),
        // Usage
        {
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          {
            VMA_MEMORY_USAGE_GPU_ONLY,
            0u,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          },
        },
      }),
    // Deleter
    std::bind(release_buffer, std::ref(pool_), std::placeholders::_1),
  };
}

Persistent::Image Persistent::Pool::image(
    const VkExtent3D& extents,
    const TensorOptions& options) {
  return {
    // Resource
    pool_.image({
        VK_IMAGE_TYPE_3D,
        api::utils::convert(options.dtype()),
        extents,
        // Usage
        {
          VK_IMAGE_USAGE_SAMPLED_BIT |
              VK_IMAGE_USAGE_STORAGE_BIT,
          {
            VMA_MEMORY_USAGE_GPU_ONLY,
            0u,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          },
        },
        // View
        {
          VK_IMAGE_VIEW_TYPE_3D,
          api::utils::convert(options.dtype()),
        },
      }),
    // Deleter
    std::bind(release_image, std::ref(pool_), std::placeholders::_1),
  };
}

Persistent::Pool* persistent() {
  typedef Persistent::Pool Pool;

  static const std::unique_ptr<Pool> pool([]() -> Pool* {
    try {
      return new Pool(api::context()->gpu());
    }
    catch (...) {
      return nullptr;
    }
  }());

  TORCH_CHECK(
      pool,
      "Vulkan: Failed to initialize the persistent resource pool!");

  return pool.get();
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
