#include <ATen/native/vulkan/graph/Graph.h>
#include <ATen/native/vulkan/graph/Staging.h>

namespace at {
namespace native {
namespace vulkan {

//
// SharedObject
//

void SharedObject::add_user(ComputeGraph* const graph, const ValueRef idx) {
  vTensor& t = graph->get_val(idx).toTensor();

  //
  // Aggregate Memory Requirements
  //

  const VkMemoryRequirements mem_reqs = t.get_memory_requirements();
  aggregate_memory_requirements.size =
      std::max(mem_reqs.size, aggregate_memory_requirements.size);
  aggregate_memory_requirements.alignment =
      std::max(mem_reqs.alignment, aggregate_memory_requirements.alignment);
  aggregate_memory_requirements.memoryTypeBits |= mem_reqs.memoryTypeBits;

  //
  // Aggregate Allocation Create Info
  //

  const VmaAllocationCreateInfo create_info = t.get_allocation_create_info();
  // Clear out CREATE_STRATEGY bit flags in case of conflict
  VmaAllocationCreateFlags clear_mask = ~VMA_ALLOCATION_CREATE_STRATEGY_MASK;
  VmaAllocationCreateFlags create_flags = create_info.flags & clear_mask;
  // Use the default allocation strategy
  aggregate_create_info.flags = create_flags | api::DEFAULT_ALLOCATION_STRATEGY;

  // Set the usage flag if it is currently not set
  if (aggregate_create_info.usage == VMA_MEMORY_USAGE_UNKNOWN) {
    aggregate_create_info.usage = create_info.usage;
  }
  // Otherwise check that there is no conflict regarding usage
  VK_CHECK_COND(aggregate_create_info.usage == create_info.usage);
  aggregate_create_info.requiredFlags |= create_info.requiredFlags;
  aggregate_create_info.preferredFlags |= create_info.preferredFlags;

  users.emplace_back(idx);
}

void SharedObject::allocate(ComputeGraph* const graph) {
  if (aggregate_memory_requirements.size == 0) {
    return;
  }
  allocation = graph->context()->adapter_ptr()->vma().create_allocation(
      aggregate_memory_requirements, aggregate_create_info);
}

void SharedObject::bind_users(ComputeGraph* const graph) {
  if (users.empty()) {
    return;
  }
  for (const ValueRef idx : users) {
    graph->get_val(idx).toTensor().bind_allocation(allocation);
  }
}

//
// ComputeGraph
//

ComputeGraph::ComputeGraph(GraphConfig config)
    : config_{config},
      context_{new api::Context(
          api::runtime()->default_adapter_i(),
          config_.contextConfig)},
      shared_objects_{},
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
    const api::ScalarType dtype,
    const int64_t shared_object_idx) {
  bool allocate_memory = shared_object_idx < 0;

  ValueRef idx(static_cast<int>(values_.size()));
  values_.emplace_back(vTensor(
      context(),
      sizes,
      dtype,
      api::StorageType::TEXTURE_3D,
      api::GPUMemoryLayout::TENSOR_CHANNELS_PACKED,
      allocate_memory));

  if (!allocate_memory) {
    get_shared_object(shared_object_idx).add_user(this, idx);
  }
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

SharedObject& ComputeGraph::get_shared_object(const int64_t idx) {
  if (idx >= shared_objects_.size()) {
    shared_objects_.resize(idx + 1);
  }
  return shared_objects_[idx];
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

  context_->flush();
}

void ComputeGraph::encode_execute() {
  context_->flush();
  context_->set_cmd(/*reusable = */ true);

  for (SharedObject& shared_object : shared_objects_) {
    shared_object.allocate(this);
    shared_object.bind_users(this);
  }

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
