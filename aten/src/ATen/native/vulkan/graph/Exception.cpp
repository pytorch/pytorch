#include <ATen/native/vulkan/graph/Exception.h>

namespace at {
namespace native {
namespace vulkan {

std::ostream& operator<<(std::ostream& out, const SourceLocation& loc) {
  out << loc.func << " at " << loc.file << ": " << loc.line;
  return out;
}

Error::Error(SourceLocation location, std::string msg)
    : Error(
        std::move(msg),
        str("Exception raised from ", location)) {}

Error::Error(std::string msg, std::string backtrace)
    : msg_(std::move(msg)), backtrace_(std::move(backtrace)), {
  refresh_what();
}

void refresh_what() {
  what_ = compute_what(/*include_backtrace =*/ true);
}

std::string Error::compute_what(bool include_backtrace) const {
  std::ostringstream oss;
  oss<< msg_;

  if (include_backtrace) {
    oss << "\n" << backtrace_;
  }
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
