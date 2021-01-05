#pragma once

#ifdef USE_VULKAN_API

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
    Buffer(VkCommandBuffer command_buffer = VK_NULL_HANDLE);
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&);
    Buffer& operator=(Buffer&&);
    ~Buffer() = default;

    operator bool() const;

    void begin();
    void end();
    void barrier(const Pipeline::Barrier& barrier);
    void bind(const Pipeline::Object& pipeline);
    void bind(const Descriptor::Set& set);
    void copy(Resource::Buffer::Object source, Resource::Buffer::Object destination);
    void dispatch(const Shader::WorkGroup& global_work_group);
    void submit(VkQueue queue, Resource::Fence fence = {});

   private:
    void barrier();

   private:
    VkCommandBuffer command_buffer_;

    struct Bound final {
      Pipeline::Object pipeline;
      VkDescriptorSet descriptor_set;

      void reset();
    } bound_;

    struct Barrier final {
      struct Stage final {
        VkPipelineStageFlags src;
        VkPipelineStageFlags dst;

        operator bool() const;
      } stage;

      c10::SmallVector<Resource::Buffer::Barrier, 4u> buffers;
      c10::SmallVector<Resource::Image::Barrier, 4u> images;

      void reset();
    } barriers_;
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
    ~Pool();

    Buffer allocate();
    void purge();

   private:
    struct Configuration final {
      static constexpr uint32_t kQuantum = 64u;
      static constexpr uint32_t kReserve = 1024u;
    };

    VkDevice device_;
    Handle<VkCommandPool, VK_DELETER(CommandPool)> command_pool_;

    struct {
      std::vector<VkCommandBuffer> pool;
      size_t in_use;
    } buffer_;
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
    bound_(std::move(buffer.bound_)),
    barriers_(std::move(buffer.barriers_)) {
  buffer.command_buffer_ = VK_NULL_HANDLE;
}

inline Command::Buffer& Command::Buffer::operator=(Buffer&& buffer) {
  if (&buffer != this) {
    command_buffer_ = std::move(buffer.command_buffer_);
    bound_ = std::move(buffer.bound_);
    barriers_ = std::move(buffer.barriers_);

    buffer.command_buffer_ = VK_NULL_HANDLE;
  };

  return *this;
}

inline Command::Buffer::operator bool() const {
  return VK_NULL_HANDLE != command_buffer_;
}

inline void Command::Buffer::Bound::reset() {
  pipeline = {};
  descriptor_set = VK_NULL_HANDLE;
}

inline Command::Buffer::Barrier::Stage::operator bool() const {
  return (0u != src) ||
         (0u != dst);
}

inline void Command::Buffer::Barrier::reset() {
  stage = {};
  buffers.clear();
  images.clear();
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
