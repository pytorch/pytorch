#include <ATen/native/vulkan/impl/Packing.h>

#include <ATen/native/vulkan/graph/Staging.h>

namespace at {
namespace native {
namespace vulkan {

void TensorStaging::memcpy_to_mapping(void* src, api::MemoryMap& dst_mapping) {
  size_t nbytes = tensor.nbytes();
  if (tensor.dtype() == at::kFloat) {
    memcpy_to_mapping_impl<float>(src, dst_mapping, nbytes);
  } else if (tensor.dtype() == at::kHalf) {
    memcpy_to_mapping_impl<c10::Half>(src, dst_mapping, nbytes);
  } else if (tensor.dtype() == c10::kQUInt8) {
    memcpy_to_mapping_impl<c10::quint8>(src, dst_mapping, nbytes);
  } else if (tensor.dtype() == c10::kQInt8) {
    memcpy_to_mapping_impl<c10::qint8>(src, dst_mapping, nbytes);
  } else if (tensor.dtype() == c10::kQInt32) {
    memcpy_to_mapping_impl<c10::qint32>(src, dst_mapping, nbytes);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " at::kHalf or at::Float but got ",
        tensor.dtype());
  }
}

void TensorStaging::memcpy_from_mapping(
    api::MemoryMap& src_mapping,
    void* dst) {
  size_t nbytes = tensor.nbytes();
  if (tensor.dtype() == at::kFloat) {
    memcpy_from_mapping_impl<float>(src_mapping, dst, nbytes);
  } else if (tensor.dtype() == at::kHalf) {
    memcpy_from_mapping_impl<c10::Half>(src_mapping, dst, nbytes);
  } else if (tensor.dtype() == c10::kQUInt8) {
    memcpy_from_mapping_impl<c10::quint8>(src_mapping, dst, nbytes);
  } else if (tensor.dtype() == c10::kQInt8) {
    memcpy_from_mapping_impl<c10::qint8>(src_mapping, dst, nbytes);
  } else if (tensor.dtype() == c10::kQInt32) {
    memcpy_from_mapping_impl<c10::qint32>(src_mapping, dst, nbytes);
  } else {
    TORCH_CHECK(
        false,
        "Invalid Data Type: expected c10::kQInt32, c10::kQInt8, c10::kQUInt8,",
        " at::kHalf or at::Float but got ",
        tensor.dtype());
  }
}

TensorStaging::TensorStaging(TensorStaging&& other) noexcept
    : tensor(std::move(other.tensor)), staging(std::move(other.staging)) {}

TensorStaging::TensorStaging(
    api::Context* context,
    IntArrayRef sizes,
    c10::ScalarType dtype,
    api::StorageType storage_type,
    c10::MemoryFormat memory_format)
    : tensor{
        context,
        sizes,
        dtype,
        storage_type,
        memory_format,
      },
      staging{std::make_unique<api::StorageBuffer>(
        context,
        dtype,
        tensor.gpu_numel())} {}

void TensorStaging::ptr_to_staging(void* src) {
  api::MemoryMap mapping(staging->buffer(), api::MemoryAccessType::WRITE);
  mapping.invalidate();

  memcpy_to_mapping(src, mapping);
}

void TensorStaging::staging_to_ptr(void* dst) {
  api::MemoryMap mapping(staging->buffer(), api::MemoryAccessType::READ);
  mapping.invalidate();

  memcpy_from_mapping(mapping, dst);
}

void TensorStaging::record_copy_to_gpu(api::Context* context) {
  api::ShaderInfo shader = packing::get_nchw_to_image_shader(tensor);

  api::PipelineBarrier pipeline_barrier{};
  packing::record_nchw_to_image_op(
      context,
      shader,
      staging->buffer(),
      tensor,
      pipeline_barrier,
      VK_NULL_HANDLE);
}

void TensorStaging::record_copy_from_gpu(api::Context* context) {
  api::ShaderInfo shader = packing::get_image_to_nchw_shader(tensor);

  api::PipelineBarrier pipeline_barrier{};
  packing::record_image_to_nchw_op(
      context,
      shader,
      tensor,
      staging->buffer(),
      pipeline_barrier,
      VK_NULL_HANDLE);
}

} // namespace vulkan
} // namespace native
} // namespace at
