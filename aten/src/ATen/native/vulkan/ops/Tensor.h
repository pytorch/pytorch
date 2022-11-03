#pragma once

#ifdef USE_VULKAN_API

#include <ATen/core/Tensor.h>
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
      const api::StorageType storage_type,
      const TensorOptions& options);
  vTensorStorage(
      api::Context* context,
      IntArrayRef sizes,
      const api::StorageType storage_type,
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
  api::StorageType storage_type_;
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

 public:
  inline VkFormat texture_format() {
    return image_.format();
  }
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
      api::Context* context,
      IntArrayRef sizes,
      const api::StorageType storage_type,
      const TensorOptions& options);

  vTensor(
      api::Context* const context,
      const IntArrayRef sizes,
      const TensorOptions& options,
      double q_scale,
      int64_t q_zero_point);

  vTensor(
      api::Context* const context,
      const IntArrayRef sizes,
      const api::StorageType storage_type,
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

  inline api::StorageType storage_type() const {
    return view_->storage_type_;
  }

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

  /*
   * Get a c10::ScalarType that corresponds to the image format of the texture
   */
  inline c10::ScalarType texture_dtype() const {
    return api::c10_scalartype(view_->texture_format());
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

  inline void set_is_quantized() const {
    view_->is_quantized_ = true;
  }

  inline bool is_quantized() const {
    return view_->is_quantized_;
  }

  inline void set_scale(const double q_scale) const {
    view_->q_scale = q_scale;
  }

  inline double get_scale() const {
    return view_->q_scale;
  }

  inline float get_scale_float() const {
    return api::utils::safe_downcast<float>(view_->q_scale);
  }

  inline void set_zero_point(const int64_t q_zero_point) const {
    view_->q_zero_point = q_zero_point;
  }

  inline int64_t get_zero_point() const {
    return view_->q_zero_point;
  }

  inline int32_t get_zero_point_int32() const {
    return api::utils::safe_downcast<int32_t>(view_->q_zero_point);
  }

  inline size_t nbytes() const {
    return c10::elementSize(c10::typeMetaToScalarType(options().dtype())) *
        c10::multiply_integers(sizes());
  }

  /*
   * Number of texels in the image texture.
   */
  inline VkDeviceSize numtexels() {
    return view_->extents_.data[0u] * view_->extents_.data[1u] *
        view_->extents_.data[2u];
  }

  /*
   * Number of "cells" in the image texture. 4 cells make up a texel.
   */
  inline VkDeviceSize numcells() {
    return view_->extents_.data[0u] * view_->extents_.data[1u] *
        (4u * view_->extents_.data[2u]);
  }

  /*
   * Number of bytes needed for a buffer to receive all data in the texture
   */
  inline VkDeviceSize buffer_bytes() {
    return c10::elementSize(this->texture_dtype()) * view_->extents_.data[0u] *
        view_->extents_.data[1u] * (4u * view_->extents_.data[2u]);
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
