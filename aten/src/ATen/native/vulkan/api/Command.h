#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

class CommandBuffer final {
 public:
  explicit CommandBuffer(
      const VkCommandBuffer,
      const VkCommandBufferUsageFlags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  CommandBuffer(CommandBuffer&&) noexcept;
  CommandBuffer& operator=(CommandBuffer&&) noexcept;

  ~CommandBuffer() = default;

  // The lifecycle of a command buffer is as follows:
  enum State {
    INVALID,  // Used to indicate the command buffer is moved from
    NEW,  // Set during constructor
    RECORDING,  // Set during call to begin(), dispatch(), and copy_texture_to_texture()
    PIPELINE_BOUND,  // Set during call to  bind_pipeline()
    DESCRIPTORS_BOUND,  // Set during call to bind_descriptors()
    BARRIERS_INSERTED,  // Set during call to insert_barrier()
    READY,  //  Set during call to end()
    SUBMITTED,  // Set during call to get_submit_handle()
  };

  struct Bound {
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    utils::uvec3 local_workgroup_size;
    VkDescriptorSet descriptors;

    explicit Bound()
      : pipeline{VK_NULL_HANDLE},
        pipeline_layout{VK_NULL_HANDLE},
        local_workgroup_size{0u, 0u, 0u},
        descriptors{VK_NULL_HANDLE} {
    }

    inline void reset() {
      pipeline = VK_NULL_HANDLE;
      pipeline_layout = VK_NULL_HANDLE;
      local_workgroup_size = {0u, 0u, 0u};
      descriptors = VK_NULL_HANDLE;
    }
  };

 private:
  VkCommandBuffer handle_;
  VkCommandBufferUsageFlags flags_;
  State state_;
  Bound bound_;

 public:
  void begin();
  void end();

  void bind_pipeline(
      const VkPipeline,
      const VkPipelineLayout,
      const utils::uvec3);
  void bind_descriptors(const VkDescriptorSet);

  void insert_barrier(const PipelineBarrier& pipeline_barrier);
  void dispatch(const utils::uvec3&);

  void copy_texture_to_texture(
      const api::VulkanImage&,
      const api::VulkanImage&,
      const api::utils::uvec3&,
      const api::utils::uvec3&,
      const api::utils::uvec3&);

  VkCommandBuffer get_submit_handle();

  inline operator bool() const {
    return VK_NULL_HANDLE != handle_;
  }
};

struct CommandPoolConfig final {
  uint32_t cmdPoolInitialSize;
  uint32_t cmdPoolBatchSize;
};

class CommandPool final {
 public:
  explicit CommandPool(
      const VkDevice, const uint32_t, const CommandPoolConfig&);

  CommandPool(const CommandPool&) = delete;
  CommandPool& operator=(const CommandPool&) = delete;

  CommandPool(CommandPool&&) noexcept;
  CommandPool& operator=(CommandPool&&) = delete;

  ~CommandPool();

 private:
  VkDevice device_;
  uint32_t queue_family_idx_;
  VkCommandPool pool_;
  CommandPoolConfig config_;
  // New Buffers
  std::vector<VkCommandBuffer> buffers_;
  size_t in_use_;

 public:
  CommandBuffer get_new_cmd();

  void flush();

 private:
  void allocate_new_batch(const uint32_t);
};

struct Command final {
  class Pool;

  //
  // Buffer
  //

  class Buffer final {
   public:
    explicit Buffer(VkCommandBuffer command_buffer = VK_NULL_HANDLE);
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&);
    Buffer& operator=(Buffer&&);
    ~Buffer() = default;

    operator bool() const;
    VkCommandBuffer handle() const;

    void begin();
    void end();

    void barrier(const PipelineBarrier& barrier);
    void bind(
        const VkPipeline pipeline,
        VkPipelineLayout pipeline_layout,
        utils::uvec3 local_work_group);
    void bind(const Descriptor::Set& set);
    void copy(
        const api::VulkanBuffer::Package source,
        const api::VulkanBuffer::Package destination);
    void copy_image(
        const api::VulkanImage&,
        const api::VulkanImage&,
        const api::utils::uvec3&,
        const api::utils::uvec3&,
        const api::utils::uvec3&);
    void dispatch(const utils::uvec3& global_work_group);

   private:
    friend class Pool;

    void barrier();
    void invalidate();

   private:
    VkCommandBuffer command_buffer_;

    struct Bound final {
      VkPipeline pipeline;
      VkPipelineLayout pipeline_layout;
      utils::uvec3 local_work_group;
      VkDescriptorSet descriptor_set;

      void reset();
    } bound_;

    struct Barrier final {
      struct Stages final {
        VkPipelineStageFlags src;
        VkPipelineStageFlags dst;

        operator bool() const;
      } stage;

      c10::SmallVector<api::BufferMemoryBarrier, 4u> buffer_barriers;
      c10::SmallVector<api::ImageMemoryBarrier, 4u> image_barriers;

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
    Buffer& stream();
    void purge();

    void submit(
        VkQueue queue,
        c10::ArrayRef<const Buffer> buffers,
        const VkFence fence = VK_NULL_HANDLE);

   private:
    void invalidate();

   private:
    struct Configuration final {
      static constexpr uint32_t kQuantum = 4u;
      static constexpr uint32_t kReserve = 16u;
      static constexpr uint32_t kSubmit = 16u;
    };

    VkDevice device_;
    Handle<VkCommandPool, VK_DELETER(CommandPool)> command_pool_;

    struct {
      std::vector<VkCommandBuffer> pool;
      size_t in_use;
    } buffer_;

    struct {
      Buffer buffer;
      uint32_t counter;
    } stream_;
  } pool /* [thread_count] */;

  explicit Command(const GPU& gpu)
    : pool(gpu) {
  }
};

//
// Impl
//

inline Command::Buffer::operator bool() const {
  return VK_NULL_HANDLE != command_buffer_;
}

inline VkCommandBuffer Command::Buffer::handle() const {
  return command_buffer_;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
