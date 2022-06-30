#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ATen.h>
#include <ATen/native/vulkan/api/api.h>
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class StorageState final {
 public:
  StorageState();
  StorageState(const api::Adapter*, IntArrayRef);

  struct Bundle final {
    struct Buffer final {
      VkPipelineStageFlags stage;
      VkAccessFlags access;

      operator bool() const;
    } staging, buffer;

    struct Image final {
      VkPipelineStageFlags stage;
      VkAccessFlags access;
      VkImageLayout layout;

      operator bool() const;
    } image;
  };

  struct Component final {
    typedef uint8_t Flags;

    enum Type : Flags {
      Buffer = 1u << 0u,
      Image = 1u << 1u,
      Staging = 1u << 2u,
      All = Buffer | Image | Staging,
    };
  };

  // Availability
  bool is_available(Component::Flags) const;
  bool is_discrete() const;
  bool is_uma() const;

  // Clean / Dirty
  bool is_clean(Component::Flags) const;
  bool is_dirty(Component::Flags) const;
  void set_clean(Component::Flags);
  void set_dirty(Component::Flags);

  // Transition
  typedef std::pair<Bundle, Bundle> Transition;
  Transition transition(Bundle to);

 private:
  Component::Flags available_;
  Component::Flags dirty_;
  Bundle bundle_;

 private:
 #ifdef VULKAN_TENSOR_DEBUG
  friend class vTensorStorage;
 #endif /* VULKAN_TENSOR_DEBUG */
};


class vTensorStorage final {
 public:
  vTensorStorage();

  vTensorStorage(
      api::Context* context,
      IntArrayRef sizes,
      const TensorOptions& options);

  vTensorStorage(const vTensorStorage&) = delete;
  vTensorStorage& operator=(const vTensorStorage&) = delete;

  vTensorStorage(vTensorStorage&&) = default;
  vTensorStorage operator=(vTensorStorage&&) = delete;

  ~vTensorStorage();

  friend class vTensor;

 private:

  class CMD;

  typedef StorageState::Component Component;

 private:
  // Resources
  mutable api::VulkanBuffer buffer_;
  mutable api::VulkanImage image_;
  mutable api::VulkanBuffer staging_;
  mutable api::VulkanFence fence_;

  // Context
  api::Context* context_;

  // StorageState
  mutable StorageState state_;

  // Metadata
  api::utils::uvec3 extents_;
  TensorOptions options_;
  c10::SmallVector<int64_t, 6u> sizes_;
  c10::SmallVector<int64_t, 6u> strides_;

 public:
  void release();

  api::VulkanBuffer& buffer(
      api::Command::Buffer&, api::PipelineStageFlags, api::MemoryAccessFlags) const;

  api::VulkanImage& image(
      api::Command::Buffer&, api::PipelineStageFlags, api::MemoryAccessFlags) const;

  api::VulkanBuffer& staging(
      api::Command::Buffer&, api::PipelineStageFlags, api::MemoryAccessFlags) const;

  void wait_for_fence() const;

  /*
    Metadata
  */

  inline bool has_image() const {
    return state_.is_available(vTensorStorage::Component::Image);
  }

  inline const api::utils::uvec3& extents() const {
    return extents_;
  }

  inline const TensorOptions& options() const {
    return options_;
  }

  inline IntArrayRef sizes() const {
    return sizes_;
  }

  inline IntArrayRef strides() const {
    return strides_;
  }

 private:
  // Accessors / Lazy Allocation
  api::VulkanBuffer& buffer() const;
  api::VulkanBuffer& buffer(CMD&, api::PipelineStageFlags, api::MemoryAccessFlags) const;
  api::VulkanImage& image() const;
  api::VulkanImage& image(CMD&, api::PipelineStageFlags, api::MemoryAccessFlags) const;
  api::VulkanBuffer& staging() const;
  api::VulkanBuffer& staging(CMD&, api::PipelineStageFlags, api::MemoryAccessFlags) const;
  api::VulkanFence& fence(api::MemoryAccessFlags) const;

  // Validation
  void verify() const;
};

class vTensor final {
 public:
  vTensor() = default;
  vTensor(
      api::Context* context,
      IntArrayRef sizes,
      const TensorOptions& options);

  // Even at the cost of a heap allocation plus the resulting negative impact
  // on cache locality due to the subsequent pointer chasing, it is still
  // critcal to share the view across vTensor implementations to minimize
  // programmer errors.  Ideally this class should have been only made movable,
  // and non-copyable - something we cannot do unfortunately due to the inner
  // workings of at::TensorImpl requiring copy semantics in
  // at::TensorImpl::release_resources() to function as expected.  Now that this
  // class is made copyable though, a new door to a whole new class of bugs is
  // opened, in that there now is a chance of two [shallow] copies, have their
  // StorageState objects go out of sync as a result of an operation being performed on
  // one shallow copy that is not reflected in the other.  Technically, if the
  // programmer is very careful, it is possible to avoid this trap and not pay
  // the cost of indirection, but the resulting bugs of missing memory barriers
  // will be so frustrating to hunt down for those unfamiliar with the internal
  // mechanics of this class, that I decided to take the performance pentalty
  // of this extra layer of indirection in favor of making this class easier
  // to use.

 private:
  std::shared_ptr<vTensorStorage> view_;

 public:

  // Host access

  inline void wait_for_fence() const {
    view_->wait_for_fence();
  }

  inline api::VulkanBuffer& host_buffer(
      api::Command::Buffer& command_buffer,
      const api::MemoryAccessFlags access) & {
    return view_->staging(command_buffer, api::PipelineStage::Host, access);
  }

  // Resource access

  inline api::VulkanBuffer::Package buffer(
      api::Command::Buffer& command_buffer,
      api::PipelineStageFlags stage) const & {
    return view_->buffer(
        command_buffer,
        stage,
        api::MemoryAccessType::READ).package();
  }

  inline api::VulkanBuffer::Package buffer(
      api::Command::Buffer& command_buffer,
      api::PipelineStageFlags stage,
      api::MemoryAccessFlags access) & {
    return view_->buffer(
        command_buffer,
        stage,
        access).package();
  }

  inline bool has_image() const {
    return view_->has_image();
  }

  api::VulkanImage::Package image(
      api::Command::Buffer& command_buffer,
      api::PipelineStageFlags stage) const & {
    return view_->image(
        command_buffer,
        stage,
        api::MemoryAccessType::READ).package();
  }

  api::VulkanImage::Package image(
      api::Command::Buffer& command_buffer,
      api::PipelineStageFlags stage,
      api::MemoryAccessFlags access) & {
    return view_->image(
        command_buffer,
        stage,
        access).package();
  }

  /*
    Metadata
  */

  inline const api::utils::uvec3& extents() const {
    return view_->extents_;
  }

  inline const TensorOptions& options() const {
    return view_->options_;
  }

  inline IntArrayRef sizes() const {
    return view_->sizes_;
  }

  inline IntArrayRef strides() const {
    return view_->strides_;
  }

  inline size_t nbytes() const {
    return c10::elementSize(c10::typeMetaToScalarType(options().dtype()))
           * c10::multiply_integers(sizes());
  }

 private:

 private:
 #ifdef VULKAN_TENSOR_DEBUG
  friend std::ostream& operator<<(
      std::ostream&,
      const StorageState::Bundle&);
 #endif /* VULKAN_TENSOR_DEBUG */
};

vTensor& convert(const Tensor& tensor);
Tensor convert(const vTensor& tensor);

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;
void verify(const TensorOptions& options);

//
// Impl
//

inline StorageState::Bundle::Buffer::operator bool() const {
  return (0u != stage) &&
         (0u != access);
}

inline StorageState::Bundle::Image::operator bool() const {
  return (0u != stage) &&
         (0u != access) &&
         (VK_IMAGE_LAYOUT_UNDEFINED != layout);
}

inline bool StorageState::is_available(
    const Component::Flags components) const {
  return available_ & components;
}

inline bool StorageState::is_discrete() const {
  return is_available(Component::Staging);
}

inline bool StorageState::is_uma() const {
  return !is_discrete();
}

inline bool StorageState::is_clean(
    const Component::Flags components) const {
  return !is_dirty(components);
}

inline bool StorageState::is_dirty(
    const Component::Flags components) const {
  return dirty_ & components;
}

inline void StorageState::set_clean(
    const Component::Flags components) {
  dirty_ &= ~components;
}

inline void StorageState::set_dirty(
    const Component::Flags components) {
  dirty_ |= components;
}

inline vTensor& convert(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      tensor.is_vulkan(),
      "Vulkan tensor expected!");

  vTensorImpl* const impl =
      static_cast<vTensorImpl*>(tensor.unsafeGetTensorImpl());

  return impl->unsafe_opaque_handle();
}

inline Tensor convert(const vTensor& tensor) {
  return at::detail::make_tensor<vTensorImpl>(
      DispatchKeySet(DispatchKey::Vulkan),
      tensor.options().dtype(),
      at::Device(at::kVulkan),
      tensor,
      tensor.sizes(),
      tensor.strides());
}

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
