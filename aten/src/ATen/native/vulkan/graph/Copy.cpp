#include <ATen/native/vulkan/graph/Copy.h>

namespace at {
namespace native {
namespace vulkan {

void add_copy_node(
    ComputeGraph& graph,
    const ValueRef from,
    const ValueRef to) {
  graph.execute_nodes().emplace_back(new CopyNode(from, to));
}

ValueRef add_copy_node(ComputeGraph& graph, const ValueRef from) {
  std::vector<int64_t> out_sizes = graph.get_val_sizes(from);
  api::ScalarType out_dtype = graph.get_val_dtype(from);
  ValueRef to = graph.add_tensor(out_sizes, out_dtype);
  add_copy_node(graph, from, to);
  return to;
}

CopyNode::CopyNode(const ValueRef from, const ValueRef to) {
  inputs_.emplace_back(from);
  outputs_.emplace_back(to);
}

void CopyNode::encode_execute(ComputeGraph* graph) const {
  api::PipelineBarrier pipeline_barrier{};

  vTensor& from_tensor = graph->get_val(inputs_[0]).toTensor();
  vTensor& to_tensor = graph->get_val(outputs_[0]).toTensor();

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
