#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Runtime.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

Context* initialize() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      const Adapter adapter = runtime()->select([](const Adapter& adapter) {
        // Select the first adapter.
        return true;
      });

      return new Context(adapter);
    }
    catch (...) {
      return nullptr;
    }
  }());

  return context.get();
}

VkDevice create_device(
    const VkPhysicalDevice physical_device,
    const uint32_t compute_queue_family_index) {
  const float queue_priorities = 1.0f;
  const VkDeviceQueueCreateInfo device_queue_create_info{
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    nullptr,
    0u,
    compute_queue_family_index,
    1u,
    &queue_priorities,
  };

  const VkDeviceCreateInfo device_create_info{
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    nullptr,
    0u,
    1u,
    &device_queue_create_info,
    0u,
    nullptr,
    0u,
    nullptr,
  };

  VkDevice device{};
  VK_CHECK(vkCreateDevice(physical_device, &device_create_info, nullptr, &device));

  return device;
}

VkQueue acquire_queue(
    const VkDevice device,
    const uint32_t compute_queue_family_index) {
  VkQueue queue{};
  vkGetDeviceQueue(device, compute_queue_family_index, 0, &queue);
  return queue;
}

} // namespace

Context::Context(const Adapter& adapter)
    : adapter_(adapter),
      device_(
          create_device(
              adapter.physical_device,
              adapter.compute_queue_family_index),
          &VK_DELETER(Device)),
      queue_(acquire_queue(device(), adapter.compute_queue_family_index)),
      command_(device(), {adapter.compute_queue_family_index}),
      shader_(device()),
      pipeline_(device()),
      descriptor_(device()),
      resource_(adapter.runtime->instance(), adapter.physical_device, device()) {
}

Context* context() {
  Context* const context = initialize();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(context);

  return context;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
