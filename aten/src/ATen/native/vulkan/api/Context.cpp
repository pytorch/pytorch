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
      shader_(gpu()),
      pipeline_(gpu()),
      descriptor_(gpu()),
      resource_(gpu()) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");
}

Context::~Context() {
  try {
    flush();
  }
  catch (...) {
  }
}

void Context::flush() {
  VK_CHECK(vkDeviceWaitIdle(device()));

  resource().pool.purge();
  descriptor().pool.purge();
  command().pool.purge();
}

bool available() {
  return context();
}

Context* context() {
  Context* const context = initialize();
  TORCH_CHECK(context, "Vulkan: Backend not available on this platform!");

  return context;
}

Descriptor::Set dispatch_prologue(
    Command::Buffer& command_buffer,
    const Shader::Layout::Signature& shader_layout_signature,
    const Shader::Descriptor& shader_descriptor,
    const Shader::WorkGroup& local_work_group) {
  Descriptor& descriptor = context()->descriptor();
  Pipeline& pipeline = context()->pipeline();
  Shader& shader = context()->shader();

  const Shader::Layout::Object shader_layout =
      shader.layout.cache.retrieve({
        shader_layout_signature,
      });

  command_buffer.bind(
      pipeline.cache.retrieve({
        pipeline.layout.cache.retrieve({
          shader_layout.handle,
        }),
        shader.cache.retrieve(shader_descriptor),
        local_work_group,
      }));

  return descriptor.pool.allocate(shader_layout);
}

void dispatch_epilogue(
    Command::Buffer& command_buffer,
    const Descriptor::Set& descriptor_set,
    const Shader::WorkGroup& global_work_group) {
  command_buffer.bind(descriptor_set);
  command_buffer.dispatch(global_work_group);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
