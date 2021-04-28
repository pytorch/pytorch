#include <ATen/native/vulkan/api/Context.h>
#include <ATen/vulkan/Context.h>
#include <ATen/native/vulkan/ops/Copy.h>

#include <sstream>

namespace at {
namespace native {
namespace vulkan {
namespace api {
namespace {

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

  uint32_t device_extension_properties_count = 0;
  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device,
      nullptr,
      &device_extension_properties_count,
      nullptr));

  std::vector<VkExtensionProperties> device_extension_properties(
      device_extension_properties_count);

  VK_CHECK(vkEnumerateDeviceExtensionProperties(
      physical_device,
      nullptr,
      &device_extension_properties_count,
      device_extension_properties.data()));

  constexpr const char* const requested_device_extensions[]{
  #ifdef VK_KHR_portability_subset
    // https://vulkan.lunarg.com/doc/view/1.2.162.0/mac/1.2-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pProperties-04451
    VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
  #endif
  };

  std::vector<const char*> enabled_device_extensions;

  for (const auto& requested_device_extension : requested_device_extensions) {
    for (const auto& extension : device_extension_properties) {
      if (strcmp(requested_device_extension, extension.extensionName) == 0) {
        enabled_device_extensions.push_back(requested_device_extension);
        break;
      }
    }
  }

  const VkDeviceCreateInfo device_create_info{
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    nullptr,
    0u,
    1u,
    &device_queue_create_info,
    0u,
    nullptr,
    static_cast<uint32_t>(enabled_device_extensions.size()),
    enabled_device_extensions.data(),
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
      resource_(gpu()),
      playground_cache(gpu()),
      conv2d_dw_cache(gpu()),
      conv2d_pw_cache(gpu()),
      conv2d_cache(gpu()),
      upsample_cache(gpu()) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      device_,
      "Invalid Vulkan device!");
}

Context::~Context() {
  try {
    flush();
  }
  catch (const std::exception& e) {
    TORCH_WARN(
        "Vulkan: Context destructor raised an exception! Error: ",
        e.what());
  }
  catch (...) {
    TORCH_WARN(
        "Vulkan: Context destructor raised an exception! "
        "Error: Unknown");
  }
}

void Context::flush() {
  VK_CHECK(vkQueueWaitIdle(queue()));

  resource().pool.purge();
  descriptor().pool.purge();
  command().pool.purge();
}

bool available() {
  return context();
}

Context* context() {
  static const std::unique_ptr<Context> context([]() -> Context* {
    try {
      const Adapter adapter = runtime()->select([](const Adapter& adapter) {
        // Select the first adapter.
        return true;
      });

      return new Context(adapter);
    }
    catch (const std::exception& e) {
      TORCH_WARN("Vulkan: Failed to initialize context! Error: ", e.what());
    }
    catch (...) {
      TORCH_WARN("Vulkan: Failed to initialize context! Error: Unknown");
    }

    return nullptr;
  }());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      context,
      "Invalid Vulkan context!");

  return context.get();
}

struct VulkanImpl final : public at::vulkan::VulkanImplInterface {
  bool is_vulkan_available() const override {
    return available();
  }

  Tensor& vulkan_copy_(Tensor& self, const Tensor& src) const override {
    return vulkan::ops::copy_(self, src);
  }
};
static at::vulkan::VulkanImplRegistrar g_vulkan_impl(new VulkanImpl());

Descriptor::Set dispatch_prologue(
    Command::Buffer& command_buffer,
    const Shader::Layout::Signature& shader_layout_signature,
    const Shader::Descriptor& shader_descriptor,
    const Shader::WorkGroup& local_work_group_size) {
  Context* const context = api::context();
  const GPU gpu = context->gpu();
  Descriptor& descriptor = context->descriptor();
  Pipeline& pipeline = context->pipeline();
  Shader& shader = context->shader();

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
        local_work_group_size,
      }));

  return descriptor.pool.allocate(shader_layout);
}

Descriptor::Set dispatch_prologue(
    Command::Buffer& command_buffer,
    const Context::OpCache& opcache,
    const VkDescriptorSet vk_descriptor_set) {
  Context* const context = api::context();
  const GPU gpu = context->gpu();
  Descriptor& descriptor = context->descriptor();
  Pipeline& pipeline = context->pipeline();
  Shader& shader = context->shader();

  const api::Shader::Layout::Object shader_layout =
  {
    opcache.set_layout.get(),
    opcache.layout_descriptor.signature,
  };

  const api::Pipeline::Object pipe_obj = {
    opcache.pipe.get(),
    opcache.pipe_layout.get(),
    opcache.local_work_group,
  };

  command_buffer.bind(pipe_obj);

  return Descriptor::Set(
      descriptor.pool.device_,
      vk_descriptor_set,
      shader_layout.signature);

  //return descriptor.pool.allocate(shader_layout);
}

void dispatch_epilogue(
    Command::Buffer& command_buffer,
    const Descriptor::Set& descriptor_set,
    const Shader::WorkGroup& global_work_group) {
  command_buffer.bind(descriptor_set);
  command_buffer.dispatch(global_work_group);
}

Context::OpCache::OpCache(const GPU& gpu):
  initted(false),
  set_layout{VK_NULL_HANDLE, VK_DELETER(DescriptorSetLayout)(gpu.device)},
  pipe_layout{VK_NULL_HANDLE, VK_DELETER(PipelineLayout)(gpu.device)},
  shader_module{VK_NULL_HANDLE, VK_DELETER(ShaderModule)(gpu.device)},
  pipe{VK_NULL_HANDLE, VK_DELETER(Pipeline)(gpu.device)} {
}

void Context::fill_cache(
    Context::OpCache& opcache,
    const Shader::Descriptor& shader_descriptor,
    const api::Shader::WorkGroup local_work_group) {

  //api::Descriptor::Pool& descriptor_pool = persistent()->descriptor_pool;

  opcache.set_layout = shader_.layout.cache.generate(opcache.layout_descriptor);
  opcache.pipe_layout = pipeline_.layout.cache.generate({opcache.set_layout.get()});
  opcache.shader_module = shader_.cache.generate(shader_descriptor);
  opcache.local_work_group = local_work_group;

  opcache.pipe = pipeline_.cache.generate(
    {
      opcache.pipe_layout.get(),
      opcache.shader_module.get(),
      opcache.local_work_group,
    }
  );

  const api::Shader::Layout::Object shader_layout =
  {
    opcache.set_layout.get(),
    opcache.layout_descriptor.signature,
  };

  opcache.initted = true;

  //descriptor_set = descriptor_pool.allocate_single(shader_layout);

  //api::Resource::Pool& resource_pool = persistent()->resource_pool;
}

Context::OpCache& Context::get_playground_cache() {
  if (playground_cache.initted) {
    return playground_cache;
  }
  playground_cache.layout_descriptor = {
    {
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    }
  };
  fill_cache(playground_cache, VK_KERNEL(playground), {64,1,1});
  return playground_cache;
}

Context::OpCache& Context::get_conv2d_dw_cache() {
  if (conv2d_dw_cache.initted) {
    return conv2d_dw_cache;
  }
  conv2d_dw_cache.layout_descriptor = {
    {
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    }
  };
  fill_cache(conv2d_dw_cache, VK_KERNEL(conv2d_dw), {4,4,4});
  return conv2d_dw_cache;
}

Context::OpCache& Context::get_conv2d_pw_cache() {
  if (conv2d_pw_cache.initted) {
    return conv2d_pw_cache;
  }
  conv2d_pw_cache.layout_descriptor = {
    {
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    }
  };
  fill_cache(conv2d_pw_cache, VK_KERNEL(conv2d_pw), {4,4,4});
  return conv2d_pw_cache;
}

Context::OpCache& Context::get_conv2d_cache() {
  if (conv2d_cache.initted) {
    return conv2d_cache;
  }
  conv2d_cache.layout_descriptor = {
    {
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    }
  };
  fill_cache(conv2d_cache, VK_KERNEL(conv2d), {4,4,4});
  return conv2d_cache;
}

Context::OpCache& Context::get_upsample_cache() {
  if (upsample_cache.initted) {
    return upsample_cache;
  }
  upsample_cache.layout_descriptor = {
    {
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    },
  };
  fill_cache(upsample_cache, VK_KERNEL(upsample_nearest2d), {4,4,4});
  return upsample_cache;
}
} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
