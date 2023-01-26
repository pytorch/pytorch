#include <ATen/native/vulkan/graph/Copy.h>

namespace at {
namespace native {
namespace vulkan {

CopyNode::CopyNode(ValueRef from, ValueRef to) {
  inputs_.emplace_back(from);
  outputs_.emplace_back(to);
}

void CopyNode::encode(ComputeGraph* graph) {
  api::PipelineBarrier pipeline_barrier{};

  vTensor& from_tensor = graph->tensor_at(inputs_[0]);
  vTensor& to_tensor = graph->tensor_at(outputs_[0]);

  graph->context()->submit_copy<api::VulkanImage, api::VulkanImage>(
      // pipeline barrier
      pipeline_barrier,
      // resources
      from_tensor.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::READ),
      to_tensor.image(
          pipeline_barrier,
          api::PipelineStage::TRANSFER,
          api::MemoryAccessType::WRITE),
      // copy details
      from_tensor.extents(),
      {0u, 0u, 0u},
      {0u, 0u, 0u},
      // fence handle
      VK_NULL_HANDLE);
}

} // namespace vulkan
} // namespace native
} // namespace at
