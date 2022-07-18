#pragma once

#ifdef USE_VULKAN_API

#include <ATen/ATen.h>
#include <ATen/native/vulkan/VulkanOpaqueTensorImpl.h>
#include <ATen/native/vulkan/api/api.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

struct LastAccess {
  api::PipelineStageFlags stage;
  api::MemoryAccessFlags access;

  LastAccess()
      : stage{api::PipelineStage::NO_STAGE},
        access{api::MemoryAccessType::NONE} {}

  LastAccess(
      api::PipelineStageFlags stage_flags,
      api::MemoryAccessFlags access_flags)
      : stage{stage_flags}, access{access_flags} {}
};

class vTensorStorage final {
 public:
  // Do not allow empty vTensorStorage construction
  vTensorStorage() = default;

  vTensorStorage(
      api::Context* context,
      IntArrayRef sizes,
      const TensorOptions& options);
  vTensorStorage(
      api::Context* context,
      IntArrayRef sizes,
      const TensorOptions& options,
      double q_scale,
      int64_t q_zero_point);

  vTensorStorage(const vTensorStorage&) = delete;
  vTensorStorage& operator=(const vTensorStorage&) = delete;

  vTensorStorage(vTensorStorage&&) = default;
  vTensorStorage operator=(vTensorStorage&&) = delete;

  ~vTensorStorage();

  friend class vTensor;

 private:
  // Context
  api::Context* context_;

  // Metadata
  api::utils::uvec3 extents_;
  TensorOptions options_;
  c10::SmallVector<int64_t, 6u> sizes_;
  c10::SmallVector<int64_t, 6u> strides_;
  bool is_quantized_{false};
  double q_scale{1.0f};
  int64_t q_zero_point{0u};

  // Image Texture
  mutable api::VulkanImage image_;

  // Last Access - used to insert memory barriers
  LastAccess last_access_;

 private:
  // Memory barrier insertion
  void transition(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags);

  // Validation
  void verify() const;
};

class vTensor final {
 public:
  // Do not allow empty vTensor construction
  vTensor() = default;

  vTensor(
      api::Context* context,
      IntArrayRef sizes,
      const TensorOptions& options);
  vTensor(
      api::Context* const context,
      const IntArrayRef sizes,
      const TensorOptions& options,
      double q_scale,
      int64_t q_zero_point);

 private:
  // Even at the cost of a heap allocation plus the resulting negative impact
  // on cache locality due to the subsequent pointer chasing, it is still
  // critcal to share the view across vTensor implementations to minimize
  // programmer errors.  Ideally this class should have been only made movable,
  // and non-copyable - something we cannot do unfortunately due to the inner
  // workings of at::TensorImpl requiring copy semantics in
  // at::TensorImpl::release_resources() to function as expected.  Now that this
  // class is made copyable though, a new door to a whole new class of bugs is
  // opened, in that there now is a chance of two [shallow] copies, have their
  // StorageState objects go out of sync as a result of an operation being
  // performed on one shallow copy that is not reflected in the other.
  // Technically, if the programmer is very careful, it is possible to avoid
  // this trap and not pay the cost of indirection, but the resulting bugs of
  // missing memory barriers will be so frustrating to hunt down for those
  // unfamiliar with the internal mechanics of this class, that I decided to
  // take the performance pentalty of this extra layer of indirection in favor
  // of making this class easier to use.
  std::shared_ptr<vTensorStorage> view_;

 public:
  /*
   Texture Access
  */

  api::VulkanImage& image(api::PipelineBarrier&, const api::PipelineStageFlags)
      const&;

  api::VulkanImage& image(
      api::PipelineBarrier&,
      const api::PipelineStageFlags,
      const api::MemoryAccessFlags) &;

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

  inline bool is_quantized() const {
    return view_->is_quantized_;
  }

  inline double get_scale() const {
    return view_->q_scale;
  }

  inline int64_t get_zero_point() const {
    return view_->q_zero_point;
  }

  inline size_t nbytes() const {
    return c10::elementSize(c10::typeMetaToScalarType(options().dtype())) *
        c10::multiply_integers(sizes());
  }

  inline VkDeviceSize buffer_bytes() {
    return c10::elementSize(c10::typeMetaToScalarType(options().dtype())) *
        view_->extents_.data[0u] * view_->extents_.data[1u] *
        (4u * view_->extents_.data[2u]);
  }
};

void add_buffer_barrier(
    api::PipelineBarrier&,
    const api::VulkanBuffer&,
    const api::PipelineStageFlags,
    const api::MemoryAccessFlags,
    const api::PipelineStageFlags,
    const api::MemoryAccessFlags);

using vTensorImpl = VulkanOpaqueTensorImpl<vTensor>;
void verify(const TensorOptions& options);

inline vTensor& convert(const Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(tensor.is_vulkan(), "Vulkan tensor expected!");

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

inline Tensor convert_quantized(const vTensor& tensor) {
  TORCH_CHECK(tensor.is_quantized(), "Not a Quantized Tensor");
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
