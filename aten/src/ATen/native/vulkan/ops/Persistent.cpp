#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Persistent* persistent() {
  static const std::unique_ptr<Persistent> persistent([]() -> Persistent* {
    try {
      return new Persistent{
        api::Resource::Pool{
          api::context()->gpu(),
        },
      };
    }
    catch (...) {
      return nullptr;
    }
  }());

  TORCH_CHECK(
      persistent,
      "Vulkan: Failed to initialize the persistent resource pool!");

  return persistent.get();
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
