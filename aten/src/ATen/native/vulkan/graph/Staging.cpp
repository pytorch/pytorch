#include <ATen/native/vulkan/impl/Packing.h>

#include <ATen/native/vulkan/graph/Exception.h>
#include <ATen/native/vulkan/graph/Staging.h>

namespace at {
namespace native {
namespace vulkan {

void memcpy_to_mapping(
    const void* src,
    api::MemoryMap& dst_mapping,
    const size_t nbytes,
    const c10::ScalarType dtype) {
  if (dtype == at::kFloat) {
    memcpy_to_mapping_impl<float>(src, dst_mapping, nbytes);
  } else if (dtype == at::kHalf) {
    memcpy_to_mapping_impl<c10::Half>(src, dst_mapping, nbytes);
  } else if (dtype == c10::kQUInt8) {
    memcpy_to_mapping_impl<c10::quint8>(src, dst_mapping, nbytes);
  } else if (dtype == c10::kQInt8) {
    memcpy_to_mapping_impl<c10::qint8>(src, dst_mapping, nbytes);
  } else if (dtype == c10::kQInt32) {
    memcpy_to_mapping_impl<c10::qint32>(src, dst_mapping, nbytes);
  } else {
    VKGRAPH_THROW("Unrecognized dtype!");
  }
}

void memcpy_from_mapping(
    api::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes,
    const c10::ScalarType dtype) {
  if (dtype == at::kFloat) {
    memcpy_from_mapping_impl<float>(src_mapping, dst, nbytes);
  } else if (dtype == at::kHalf) {
    memcpy_from_mapping_impl<c10::Half>(src_mapping, dst, nbytes);
  } else if (dtype == c10::kQUInt8) {
    memcpy_from_mapping_impl<c10::quint8>(src_mapping, dst, nbytes);
  } else if (dtype == c10::kQInt8) {
    memcpy_from_mapping_impl<c10::qint8>(src_mapping, dst, nbytes);
  } else if (dtype == c10::kQInt32) {
    memcpy_from_mapping_impl<c10::qint32>(src_mapping, dst, nbytes);
  } else {
    VKGRAPH_THROW("Unrecognized dtype!");
  }
}

void copy_ptr_to_staging(
    const void* src,
    api::StorageBuffer& staging,
    const size_t nbytes) {
  api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::WRITE);
  mapping.invalidate();
  memcpy_to_mapping(src, mapping, nbytes, staging.dtype());
}

void copy_staging_to_ptr(
    api::StorageBuffer& staging,
    void* dst,
    const size_t nbytes) {
  api::MemoryMap mapping(staging.buffer(), api::MemoryAccessType::READ);
  mapping.invalidate();
  memcpy_from_mapping(mapping, dst, nbytes, staging.dtype());
}

void encode_copy_to_vtensor(
    api::Context* context,
    api::StorageBuffer& staging,
    vTensor& tensor) {
  api::ShaderInfo shader = packing::get_nchw_to_image_shader(tensor);
  api::PipelineBarrier pipeline_barrier{};
  packing::record_nchw_to_image_op(
      context,
      shader,
      staging.buffer(),
      tensor,
      pipeline_barrier,
      VK_NULL_HANDLE);
}

void encode_copy_from_vtensor(
    api::Context* context,
    vTensor& tensor,
    api::StorageBuffer& staging) {
  api::ShaderInfo shader = packing::get_image_to_nchw_shader(tensor);
  api::PipelineBarrier pipeline_barrier{};
  packing::record_image_to_nchw_op(
      context,
      shader,
      tensor,
      staging.buffer(),
      pipeline_barrier,
      VK_NULL_HANDLE);
}

StagingNode::StagingNode(ValueRef from, ValueRef to) {
  inputs_.emplace_back(from);
  outputs_.emplace_back(to);
}

void StagingNode::encode_execute(ComputeGraph* graph) const {
  Value& in_val = graph->get_val(inputs_[0]);
  Value& out_val = graph->get_val(outputs_[0]);

  if (in_val.isStaging() && out_val.isTensor()) {
    api::StorageBuffer& from_staging = graph->get_val(inputs_[0]).toStaging();
    vTensor& to_tensor = graph->get_val(outputs_[0]).toTensor();
    encode_copy_to_vtensor(graph->context(), from_staging, to_tensor);
  } else if (in_val.isTensor() && out_val.isStaging()) {
    vTensor& from_tensor = graph->get_val(inputs_[0]).toTensor();
    api::StorageBuffer& to_staging = graph->get_val(outputs_[0]).toStaging();
    encode_copy_from_vtensor(graph->context(), from_tensor, to_staging);
  } else {
    VKGRAPH_THROW(
        "Unexpected input value type ",
        in_val.type(),
        " and output value type ",
        out_val.type());
  }
}

} // namespace vulkan
} // namespace native
} // namespace at
