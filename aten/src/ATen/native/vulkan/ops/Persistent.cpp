#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Pool::Pool(const api::GPU& gpu)
  : pool_(gpu) {
}

api::Resource::Buffer buffer(const c10::ArrayRef<const uint8_t> data) {
  pool_.buffer();
}

api::Resource::Image image(const c10::ArrayRef<const uint8_t> data) {
}

Pool* persistent() {
  static const std::unique_ptr<Pool> pool([]() -> Pool* {
    try {
      return new Pool(api::Context()->gpu());
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
