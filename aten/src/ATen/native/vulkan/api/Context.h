#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/QueryPool.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Runtime.h>
#include <ATen/native/vulkan/api/Shader.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Vulkan Context holds onto all relevant Vulkan state as it pertains to our
// use of Vulkan in PyTorch.  A Context is associated with one, and only one,
// Adapter as a precursor to multi-GPU support.  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
// The context is currently a global object, but technically it does not need
// to be if we were to make it explicit to the user.
//

class Context final {
 public:
  explicit Context(const VkInstance instance, size_t adapter_i);

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  Context(Context&&) = default;
  Context& operator=(Context&&) = delete;

  ~Context();

 private:
  VkInstance instance_;
  Adapter* adapter_p_;
  // Handles
  VkDevice device_;
  Adapter::Queue queue_;
  // Resource Pools
  Command command_;
  Descriptor descriptor_;
  FencePool fences_;
  QueryPool querypool_;
  // Memory Management
  std::vector<VulkanBuffer> buffers_to_clear_;
  std::vector<VulkanImage> images_to_clear_;

 public:

  inline GPU gpu() {
    const Adapter* p_adapter = adapter_p_;
    return {
      instance_,
      p_adapter,
      device_,
      queue_.family_index,
      queue_.handle,
    };
  }

  // Adapter access

  inline Adapter* adapter_ptr() {
    return adapter_p_;
  }

  // Device Caches

  inline ShaderLayoutCache& shader_layout_cache() {
    return adapter_ptr()->shader_layout_cache();
  }

  inline ShaderCache& shader_cache() {
    return adapter_ptr()->shader_cache();
  }

  inline PipelineLayoutCache& pipeline_layout_cache() {
    return adapter_ptr()->pipeline_layout_cache();
  }

  inline ComputePipelineCache& pipeline_cache() {
    return adapter_ptr()->compute_pipeline_cache();
  }

  // Resource Pools

  inline Command& command() {
    return command_;
  }

  inline Descriptor& descriptor() {
    return descriptor_;
  }

  inline FencePool& fences() {
    return fences_;
  }

  inline QueryPool& querypool() {
    return querypool_;
  }

  // Memory Management
  void register_buffer_cleanup(VulkanBuffer& buffer) {
    buffers_to_clear_.emplace_back(std::move(buffer));
  }

  void register_image_cleanup(VulkanImage& image) {
    images_to_clear_.emplace_back(std::move(image));
  }

  // GPU RPC

  template<typename... Arguments>
  void dispatch(
      Command::Buffer& command_buffer,
      const ShaderLayout::Signature& shader_layout_signature,
      const ShaderSource& shader_descriptor,
      const utils::uvec3& global_work_group,
      const utils::uvec3& local_work_group_size,
      Arguments&&... arguments);

  // This function is expensive and its use consequential for performance. Only
  // use this function for debugging or as a short term hack on way to a more
  // performant solution.

  void flush();

 private:

  inline VkDevice device() {
    return device_;
  }

  inline VkQueue queue() {
    return queue_.handle;
  }

};

class UniformParamsBuffer final {
 private:
  api::Context* context_p_;
  VulkanBuffer vulkan_buffer_;
 public:
  template<typename Block>
  UniformParamsBuffer(Context* context_p, const Block& block)
    : context_p_(context_p),
      vulkan_buffer_(
                    context_p_->adapter_ptr()->vma().create_params_buffer(block)) {
        }
  UniformParamsBuffer(const UniformParamsBuffer&) = delete;
  UniformParamsBuffer& operator=(const UniformParamsBuffer&) = delete;
  UniformParamsBuffer(UniformParamsBuffer&&) = delete;
  UniformParamsBuffer& operator=(UniformParamsBuffer&&) = delete;
  ~UniformParamsBuffer() {
      context_p_->register_buffer_cleanup(vulkan_buffer_);
    }
  VulkanBuffer& buffer() {
      return vulkan_buffer_;
    }
};

bool available();

// The global runtime is retrieved using this function, where it is declared as
// a static local variable.
Context* context();

namespace detail {

template<
    size_t...Indices,
    typename ...Arguments>
inline void bind(
    Descriptor::Set& descriptor_set,
    const std::index_sequence<Indices...>,
    Arguments&&...arguments) {
  C10_UNUSED const int _[]{
    0,
    (descriptor_set.bind(Indices, std::forward<Arguments>(arguments)), 0)...,
  };
}

} // namespace detail

template<typename... Arguments>
inline void Context::dispatch(
    Command::Buffer& command_buffer,
    const ShaderLayout::Signature& shader_layout_signature,
    const ShaderSource& shader_descriptor,
    const utils::uvec3& global_work_group,
    const utils::uvec3& local_work_group_size,
    Arguments&&... arguments) {
  // Forward declaration
  Descriptor::Set dispatch_prologue(
      Command::Buffer&,
      const ShaderLayout::Signature&,
      const ShaderSource&,
      const utils::uvec3&);

  // Factor out template parameter independent code to minimize code bloat.
  Descriptor::Set descriptor_set = dispatch_prologue(
      command_buffer,
      shader_layout_signature,
      shader_descriptor,
      local_work_group_size);

  detail::bind(
      descriptor_set,
      std::index_sequence_for<Arguments...>{},
      std::forward<Arguments>(arguments)...);

  // Forward declaration
  void dispatch_epilogue(
      Command::Buffer&,
      const Descriptor::Set&,
      const utils::uvec3&);

  // Factor out template parameter independent code to minimize code bloat.
  dispatch_epilogue(
      command_buffer,
      descriptor_set,
      global_work_group);
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
