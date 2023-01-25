#include <ATen/native/vulkan/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

ComputeGraph::ComputeGraph(GraphConfig config)
    : config_{config},
      context_{new api::Context(
          api::runtime()->default_adapter_i(),
          config_.context_config)},
      values_{},
      nodes_{} {
  context_->set_cmd(/*reusable = */ true);
}

ComputeGraph::~ComputeGraph() {
  values_.clear();
  nodes_.clear();
  context_->flush();
}

void ComputeGraph::add_input_tensor(IntArrayRef& sizes, c10::ScalarType dtype) {
  ValueRef idx(values_.size());
  values_.emplace_back(TensorStaging(context(), sizes, dtype));
  inputs_.emplace_back(idx);
}

void ComputeGraph::add_output_tensor(
    IntArrayRef& sizes,
    c10::ScalarType dtype) {
  ValueRef idx(values_.size());
  values_.emplace_back(TensorStaging(context(), sizes, dtype));
  outputs_.emplace_back(idx);
}

void ComputeGraph::copy_into_input(ValueRef idx, void* data) {
  Value& in_val = get_val(inputs_[idx]);
  in_val.toStaging().ptr_to_staging(data);
}

void ComputeGraph::copy_from_output(ValueRef idx, void* data) {
  Value& out_val = get_val(outputs_[idx]);
  out_val.toStaging().staging_to_ptr(data);
}

void ComputeGraph::encode() {
  // For each input, encode CPU to GPU transfer if it is a TensorStaging
  for (ValueRef input_ref : inputs_) {
    Value& in_val = get_val(input_ref);
    if (in_val.isStaging()) {
      in_val.toStaging().record_copy_to_gpu(context());
    }
  }

  for (std::unique_ptr<OpNode>& node : nodes_) {
    node->encode(this);
  }

  // For each output, encode GPU to CPU transfer if it is a TensorStaging
  for (ValueRef output_ref : outputs_) {
    Value& out_val = get_val(output_ref);
    if (out_val.isStaging()) {
      out_val.toStaging().record_copy_from_gpu(context());
    }
  }
}

void ComputeGraph::execute() {
  api::VulkanFence fence = context_->fences().get_fence();
  context_->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();
}

} // namespace vulkan
} // namespace native
} // namespace at
