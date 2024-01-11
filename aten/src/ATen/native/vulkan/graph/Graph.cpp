#include <ATen/native/vulkan/graph/Graph.h>
#include <ATen/native/vulkan/graph/Staging.h>

namespace at {
namespace native {
namespace vulkan {

ComputeGraph::ComputeGraph(GraphConfig config)
    : config_{config},
      context_{new api::Context(
          api::runtime()->default_adapter_i(),
          config_.contextConfig)},
      values_{},
      prepack_nodes_{},
      execute_nodes_{},
      inputs_{},
      outputs_{} {
  context_->set_cmd(/*reusable = */ true);
}

ComputeGraph::~ComputeGraph() {
  values_.clear();

  prepack_nodes_.clear();
  execute_nodes_.clear();

  context_->flush();
}

ValueRef ComputeGraph::add_tensor(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(vTensor(context(), sizes, dtype));
  return idx;
}

ValueRef ComputeGraph::add_tensorref(
    const std::vector<int64_t>& sizes,
    const api::ScalarType dtype,
    const void* const data) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(TensorRef(sizes, dtype, data));
  return idx;
}

ValueRef ComputeGraph::add_staging(
    const api::ScalarType dtype,
    const size_t numel) {
  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(api::StorageBuffer(context(), dtype, numel));
  return idx;
}

ValueRef ComputeGraph::set_input_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    vTensor& tensor = get_val(idx).toTensor();
    ValueRef staging_idx = add_staging(tensor.dtype(), tensor.gpu_numel());
    execute_nodes_.emplace_back(new StagingNode(staging_idx, idx));
    inputs_.push_back(staging_idx);
    return staging_idx;
  }
  inputs_.push_back(idx);
  return idx;
}

ValueRef ComputeGraph::set_output_tensor(
    const ValueRef idx,
    const bool use_staging) {
  if (use_staging) {
    vTensor& tensor = get_val(idx).toTensor();
    ValueRef staging_idx = add_staging(tensor.dtype(), tensor.gpu_numel());
    execute_nodes_.emplace_back(new StagingNode(idx, staging_idx));
    outputs_.push_back(staging_idx);
    return staging_idx;
  }
  outputs_.push_back(idx);
  return idx;
}

void ComputeGraph::copy_into_staging(
    const ValueRef idx,
    const void* data,
    const size_t numel) {
  Value& in_val = get_val(idx);
  api::StorageBuffer& staging = in_val.toStaging();
  size_t nbytes = numel * api::element_size(staging.dtype());
  copy_ptr_to_staging(data, staging, nbytes);
}

void ComputeGraph::copy_from_staging(
    const ValueRef idx,
    void* data,
    const size_t numel) {
  Value& out_val = get_val(idx);
  api::StorageBuffer& staging = out_val.toStaging();
  size_t nbytes = numel * api::element_size(staging.dtype());
  copy_staging_to_ptr(staging, data, nbytes);
}

void ComputeGraph::encode_prepack() {
  for (std::unique_ptr<OpNode>& node : prepack_nodes_) {
    node->encode_prepack(this);
  }
}

void ComputeGraph::prepack() const {
  // Submit and execute the command buffer
  api::VulkanFence fence = context_->fences().get_fence();
  context_->submit_cmd_to_gpu(fence.get_submit_handle(), /*final_use = */ true);
  fence.wait();

  // Flush the context and obtain a new command buffer
  context_->flush();
  context_->set_cmd(/*reusable = */ true);
}

void ComputeGraph::encode_execute() {
  for (std::unique_ptr<OpNode>& node : execute_nodes_) {
    node->encode_execute(this);
  }
}

void ComputeGraph::execute() const {
  api::VulkanFence fence = context_->fences().get_fence();
  context_->submit_cmd_to_gpu(fence.get_submit_handle());
  fence.wait();
}

} // namespace vulkan
} // namespace native
} // namespace at
