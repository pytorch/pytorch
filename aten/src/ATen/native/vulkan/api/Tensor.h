#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <c10/core/MemoryFormat.h>
#include <c10/util/accumulate.h>

namespace at {
namespace native {
namespace vulkan {

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
      const api::StorageType storage_type,
      const IntArrayRef sizes,
      const at::ScalarType dtype);

  vTensorStorage(const vTensorStorage&) = delete;
  vTensorStorage& operator=(const vTensorStorage&) = delete;

  vTensorStorage(vTensorStorage&&) = default;
  vTensorStorage operator=(vTensorStorage&&) = delete;

  ~vTensorStorage();

  friend class vTensor;

 private:
  // Context
  api::Context* context_;

  api::StorageType storage_type_;

  // Resource sizings
  api::utils::uvec3 extents_;
  int64_t buffer_length_;

  // Image Texture
  mutable api::VulkanImage image_;
  mutable api::VulkanBuffer buffer_;

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

  // Default constructor
  vTensor(
      api::Context* context,
      IntArrayRef sizes,
      const c10::ScalarType dtype = c10::kFloat,
      const api::StorageType storage_type = api::StorageType::TEXTURE_3D,
      const c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

  // Default constructor with quantization parameters
  vTensor(
      api::Context* const context,
      const IntArrayRef sizes,
      double q_scale,
      int64_t q_zero_point,
      const c10::ScalarType dtype = c10::kQUInt8,
      const api::StorageType storage_type = api::StorageType::TEXTURE_3D,
      const c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

  // Copy Constructor and Assignment; Ideally copying  would be disabled
  // (see the reasoning for move assignment below) but it is required for
  // compatibility with OpaqueTensorImpl
  vTensor(const vTensor& other) = default;
  vTensor& operator=(const vTensor& other) = default;

  // Move Constructor and assignment
  vTensor(vTensor&& other) = default;
  vTensor& operator=(vTensor&& other) = default;

  // Used for passing buffer sizes and strides data to shaders
  struct BufferMetadata {
    api::utils::uvec4 sizes;
    api::utils::uvec4 strides;
    uint32_t ndim;
    uint32_t buffer_length;
  };

 private:
  // Tensor Options
  c10::ScalarType dtype_;
  c10::MemoryFormat memory_format_;

  // Sizes and Strides
  c10::SmallVector<int64_t, 6u> sizes_;
  c10::SmallVector<int64_t, 6u> strides_;

  // Storage Dimensions. When stored on the GPU, one dimension will be aligned
  // to the next multiple of 4 in order to take advantage of vec4 data types.
  c10::SmallVector<int64_t, 6u> gpu_sizes_;
  c10::SmallVector<int64_t, 6u> gpu_strides_;

  // A Vulkan uniform buffer containing sizes and strides of the GPU buffer that
  // can be passed into a shader.
  api::UniformParamsBuffer metadata_uniform_;

  // Quantization params
  bool is_quantized_{false};
  double q_scale_{1.0f};
  int64_t q_zero_point_{0u};

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

  api::VulkanBuffer& buffer(
      api::PipelineBarrier&,
      const api::PipelineStageFlags) const&;

  api::VulkanBuffer& buffer(
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
   * Extract a ScalarType from the TensorOptions member
   */
  inline c10::ScalarType dtype() const {
    return dtype_;
  }

  /*
   * Get a c10::ScalarType that corresponds to the image format of the texture
   */
  inline c10::ScalarType texture_dtype() const {
    return api::c10_scalartype(view_->texture_format());
  }

  inline c10::MemoryFormat memory_format() const {
    return memory_format_;
  }

  inline IntArrayRef sizes() const {
    return sizes_;
  }

  inline IntArrayRef strides() const {
    return strides_;
  }

  inline IntArrayRef gpu_sizes() const {
    return gpu_sizes_;
  }

  inline IntArrayRef gpu_strides() const {
    return gpu_strides_;
  }

  /*
   * Get a uniform buffer containing sizes and strides information of the GPU
   * buffer
   */
  inline api::VulkanBuffer& buffer_metadata() {
    return metadata_uniform_.buffer();
  }

  /*
   * Constructs a BufferMetdata struct based on the original sizes and strides
   * to pass into a shader.
   */
  BufferMetadata get_cpu_buffer_metadata() const;

  inline void set_is_quantized() {
    is_quantized_ = true;
  }

  inline bool is_quantized() const {
    return is_quantized_;
  }

  inline void set_scale(const double q_scale) {
    q_scale_ = q_scale;
  }

  inline double get_scale() const {
    return q_scale_;
  }

  inline float get_scale_float() const {
    return api::utils::safe_downcast<float>(q_scale_);
  }

  inline void set_zero_point(const int64_t q_zero_point) {
    q_zero_point_ = q_zero_point;
  }

  inline int64_t get_zero_point() const {
    return q_zero_point_;
  }

  inline int32_t get_zero_point_int32() const {
    return api::utils::safe_downcast<int32_t>(q_zero_point_);
  }

  inline size_t numel() const {
    return c10::multiply_integers(sizes());
  }

  inline size_t nbytes() const {
    return c10::elementSize(dtype()) * numel();
  }

  /*
   * Returns numel but based on gpu_sizes_ instead of sizes_
   */
  inline size_t gpu_numel() const {
    return view_->buffer_length_;
  }

  /*
   * Return nbytes but bnased on gpu_sizes_ instead of sizes_
   */
  inline VkDeviceSize gpu_nbytes() const {
    return c10::elementSize(dtype()) * gpu_numel();
  }
};

void add_buffer_barrier(
    api::PipelineBarrier&,
    const api::VulkanBuffer&,
    const api::PipelineStageFlags,
    const api::MemoryAccessFlags,
    const api::PipelineStageFlags,
    const api::MemoryAccessFlags);

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
