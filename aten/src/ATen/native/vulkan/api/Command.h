#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
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
    Buffer(Buffer&&);
    Buffer& operator=(Buffer&&);
    ~Buffer() = default;

    void begin();
    void end();
    void barrier(const Pipeline::Barrier& barrier);
    void bind(const Pipeline::Object& pipeline);
    void bind(const Descriptor::Set& set);
    void copy(Resource::Buffer::Object source, Resource::Buffer::Object destination);
    void dispatch(const Shader::WorkGroup& global_work_group);
    void submit(VkQueue queue, Resource::Fence fence = {});

   private:
    VkCommandBuffer command_buffer_;
    struct {
      Pipeline::Object pipeline;
      VkDescriptorSet descriptor_set;
    } bound_;
  };

  //
  // Pool
  //

  class Pool final {
   public:
    explicit Pool(const GPU& gpu);
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    Pool(Pool&&);
    Pool& operator=(Pool&&);
    ~Pool() = default;

    Buffer allocate();
    void purge();

   private:
    VkDevice device_;
    Handle<VkCommandPool, VK_DELETER(CommandPool)> command_pool_;
  } pool /* [thread_count] */;

  explicit Command(const GPU& gpu)
    : pool(gpu) {
  }
};

//
// Impl
//

inline Command::Buffer::Buffer(Buffer&& buffer)
  : command_buffer_(std::move(buffer.command_buffer_)),
    bound_(std::move(buffer.bound_)) {
  buffer.command_buffer_ = VK_NULL_HANDLE;
}

inline Command::Buffer& Command::Buffer::operator=(Buffer&& buffer) {
  if (&buffer != this) {
    command_buffer_ = std::move(buffer.command_buffer_);
    bound_ = std::move(buffer.bound_);

    buffer.command_buffer_ = VK_NULL_HANDLE;
  };

  return *this;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
