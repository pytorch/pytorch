#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
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
  explicit Context(const Adapter& adapter);
  Context(const Context&) = delete;
  Context(Context&&) = default;
  Context& operator=(const Context&) = delete;
  Context& operator=(Context&&) = default;
  ~Context();

  GPU gpu();
  Command& command();
  Shader& shader();
  Pipeline& pipeline();
  Descriptor& descriptor();
  Resource& resource();

  // GPU RPC
  template<typename... Arguments>
  void dispatch(
      Command::Buffer& command_buffer,
      const Shader::Layout::Signature& shader_layout_signature,
      const Shader::Descriptor& shader_descriptor,
      const Shader::WorkGroup& global_work_group,
      const Shader::WorkGroup& local_work_group_size,
      Arguments&&... arguments);

  // This function is expensive and its use consequential for performance. Only
  // use this function for debugging or as a short term hack on way to a more
  // performant solution.

  void flush();

 private:
  VkDevice device();
  VkQueue queue();

 private:
  // Construction and destruction order matters.  Do not move members around.
  Adapter adapter_;
  Handle<VkDevice, decltype(&VK_DELETER(Device))> device_;
  VkQueue queue_;
  Command command_;
  Shader shader_;
  Pipeline pipeline_;
  Descriptor descriptor_;
  Resource resource_;
};

bool available();
Context* context();

//
// Impl
//

inline GPU Context::gpu() {
  // A GPU is simply a (physical device, logical device, device queue) trio.
  return {
    &adapter_,
    device(),
    queue(),
  };
}

inline Command& Context::command() {
  return command_;
}

inline Shader& Context::shader() {
  return shader_;
}

inline Pipeline& Context::pipeline() {
  return pipeline_;
}

inline Descriptor& Context::descriptor() {
  return descriptor_;
}

inline Resource& Context::resource() {
  return resource_;
}

inline VkDevice Context::device() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_);
  return device_.get();
}

inline VkQueue Context::queue() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(queue_);
  return queue_;
}

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
    const Shader::Layout::Signature& shader_layout_signature,
    const Shader::Descriptor& shader_descriptor,
    const Shader::WorkGroup& global_work_group,
    const Shader::WorkGroup& local_work_group_size,
    Arguments&&... arguments) {
  // Forward declaration
  Descriptor::Set dispatch_prologue(
      Command::Buffer&,
      const Shader::Layout::Signature&,
      const Shader::Descriptor&,
      const Shader::WorkGroup&);

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
      const Shader::WorkGroup&);

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
