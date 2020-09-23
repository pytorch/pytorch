#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Shader.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

struct Command final {
  //
  // Buffer
  //

  class Buffer final {
   public:
    Buffer(VkDevice device, VkCommandPool command_pool);
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;
    ~Buffer() = default;

    void begin();
    void end();

    void bind(VkPipeline pipeline);
    void bind(VkPipelineLayout pipeline_layout, VkDescriptorSet descriptor_set);
    void copy(VkBuffer source, VkBuffer destination, size_t size);
    void dispatch(const Shader::WorkGroup& work_group);

    void submit(VkQueue queue, VkFence fence);

   private:
    VkCommandBuffer command_buffer_;
  };

  //
  // Pool
  //

  class Pool final {
   public:
    explicit Pool(const GPU& gpu);
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&) = default;
    Pool& operator=(Pool&&) = default;
    ~Pool() = default;

    Buffer buffer();
    void purge();

   private:
    VkDevice device_;
    Handle<VkCommandPool, VK_DELETER(CommandPool)> command_pool_;
  } pool /* [thread_count] */;

  explicit Command(const GPU& gpu)
    : pool(gpu) {
  }
};

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
