#include <ATen/native/vulkan/api/Context.h>

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
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      physical_device,
      "Invalid Vulkan physical device!");

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
  TORCH_CHECK(device, "Invalid Vulkan device!");

  return device;
}

VkQueue acquire_queue(
    const VkDevice device,
    const uint32_t compute_queue_family_index) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device,
      "Invalid Vulkan device!");

  VkQueue queue{};
  vkGetDeviceQueue(device, compute_queue_family_index, 0, &queue);
  TORCH_CHECK(queue, "Invalid Vulkan queue!");

  return queue;
}

} // namespace

Context::Context(const Adapter& adapter)
    : adapter_(adapter),
      device_(
          create_device(
              adapter.handle,
              adapter.compute_queue_family_index),
          &VK_DELETER(Device)),
      queue_(acquire_queue(device(), adapter.compute_queue_family_index)),
      command_(gpu()),
      descriptor_(gpu()),
      shader_(gpu()),
      pipeline_(gpu()),
      resource_(gpu()) {
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
