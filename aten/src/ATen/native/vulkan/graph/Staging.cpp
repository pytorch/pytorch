#include <ATen/native/vulkan/impl/Packing.h>

#include <ATen/native/vulkan/graph/Staging.h>

namespace at {
namespace native {
namespace vulkan {

void memcpy_to_mapping(
    const void* src,
    api::MemoryMap& dst_mapping,
    const size_t nbytes,
    const api::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                    \
  case api::ScalarType::name:                                \
    memcpy_to_mapping_impl<ctype>(src, dst_mapping, nbytes); \
    break;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DTYPE_CASE)
    default:
      VK_THROW("Unrecognized dtype!");
  }
#undef DTYPE_CASE
}

void memcpy_from_mapping(
    api::MemoryMap& src_mapping,
    void* dst,
    const size_t nbytes,
    const api::ScalarType dtype) {
#define DTYPE_CASE(ctype, vkformat, name)                      \
  case api::ScalarType::name:                                  \
    memcpy_from_mapping_impl<ctype>(src_mapping, dst, nbytes); \
    break;

  switch (dtype) {
    VK_FORALL_SCALAR_TYPES(DTYPE_CASE)
    default:
      VK_THROW("Unrecognized dtype!");
  }
#undef DTYPE_CASE
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
    VK_THROW(
        "Unexpected input value type ",
        in_val.type(),
        " and output value type ",
        out_val.type());
  }
}

} // namespace vulkan
} // namespace native
} // namespace at
