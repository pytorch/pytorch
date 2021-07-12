#include <ATen/native/vulkan/ops/Persistent.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

Persistent* persistent() {
  static const std::unique_ptr<Persistent> persistent(
    []() -> Persistent* {
      try {
        return new Persistent{
          api::Resource::Pool{
            api::context()->gpu(),
          },
        };
      }
      catch (const std::exception& e) {
        TORCH_WARN(
            "Vulkan: Failed to initialize persistent resource pool! Error: ",
            e.what());
      }
      catch (...) {
        TORCH_WARN(
            "Vulkan: Failed to initialize persistent resource pool! "
            "Error: Unknown");
      }

      return nullptr;
    }());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      persistent,
      "Vulkan: Invalid persistent pool!");

  return persistent.get();
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at
