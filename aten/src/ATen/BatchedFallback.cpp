#include <ATen/BatchedFallback.h>
#include <ATen/VmapTransforms.h>

namespace at {

// Given a linear index, return the actual index.
// Example: Given linear_idx = 3, sizes = [5, 2], we would return [1, 0]
static SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize>
computeIndex(int64_t linear_idx, IntArrayRef sizes) {
  SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize> result;
  result.reserve(sizes.size());
  for (auto it = sizes.rbegin(); it != sizes.rend(); it++) {
    auto remainder = linear_idx % *it;
    result.push_back(remainder);
    linear_idx -= remainder;
    linear_idx /= *it;
  }
  std::reverse(std::begin(result), std::end(result));
  return result;
}

void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  auto num_returns = op.schema().returns().size();
  TORCH_CHECK(!schema.is_mutable() && !schema.hasAnyAliasInfo(),
              "Batching rule not implemented for ", schema, "; ",
              "the fallback path doesn't work on in-place or view ops.");
  TORCH_WARN("Batching rule not implemented for ", op.schema(), " falling back "
             "to slow (for loop and stack) implementation");
  TORCH_CHECK(std::all_of(op.schema().returns().begin(),
                          op.schema().returns().end(),
                          [] (const Argument& arg) { return arg.type() == TensorType::get(); }),
              "Batching rule not implemented for ", op.schema(), ". ",
              "We could not generate a fallback.");
  TORCH_CHECK(num_returns == 1,
              "Batching rule not implemented for ", op.schema(), ". ",
              "We do not yet support operations with multiple returns.");

  // Figure out which arguments are BatchedTensor. Save them to a vector.
  // For each BatchedTensor, also record what position of `stack` they came from.
  std::vector<Tensor> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (int64_t idx = 0; idx < stack->size(); ++idx) {
    const auto& ivalue = (*stack)[idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.defined()) {
      continue;
    }
    const auto* batched = maybeGetBatched(tensor);
    if (!batched) {
      continue;
    }
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(idx);
  }
  TORCH_INTERNAL_ASSERT(batched_tensor_inputs.size() > 0);

  // MultiBatchVmapTransform the BatchedTensor arguments. This returns
  // VmapPhysicalViews that contain all of the batch dimensions.
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);

  // Compute the total number of batches
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  auto some_sizes = input_physical_views.front().tensor().sizes();
  auto batch_sizes = ArrayRef<int64_t>(some_sizes.begin(), some_sizes.begin() + num_batch_dims);
  auto num_batches = std::accumulate(
      batch_sizes.begin(),
      batch_sizes.end(),
      1,
      std::multiplies<int64_t>());

  // Populate `num_batches` number of torch::jit::Stack, one for each computation.
  std::vector<torch::jit::Stack> unbatched_stacks(num_batches);
  auto pushToEachStack = [&](const auto& ivalue) {
    for (auto& stack : unbatched_stacks) {
      torch::jit::push(stack, ivalue);
    }
  };
  auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
	auto input_physical_views_iter = input_physical_views.begin();
  for (int64_t idx = 0; idx < stack->size(); ++idx) {
    const auto& ivalue = (*stack)[idx];
    if (idx != *batched_tensor_inputs_pos_iter) {
      // ivalue isn't a BatchedTensor
      pushToEachStack(ivalue);
      continue;
    }
    // ivalue is a BatchedTensor
    const auto& physical_view_for_ivalue = *input_physical_views_iter;
    for (int64_t linear_idx = 0; linear_idx < num_batches; ++linear_idx) {
      auto index = computeIndex(linear_idx, batch_sizes);
      torch::jit::push(
          unbatched_stacks[linear_idx],
          physical_view_for_ivalue.tensor().index(index));
    }
    batched_tensor_inputs_pos_iter++;
    input_physical_views_iter++;
  }

  // Call the operation once for batch
  for (auto& stack : unbatched_stacks) {
    op.callBoxed(&stack);
  }

  // Stack the tensors together to form the result.
  stack->clear();
  std::vector<Tensor> output_shards;
  for (const auto& stack : unbatched_stacks) {
    TORCH_INTERNAL_ASSERT(stack.size() == 1)
    output_shards.push_back(stack[0].toTensor());
  }
  auto flat_output = at::stack(output_shards);
  VmapDimVector output_sizes(batch_sizes);
  output_sizes.insert(
      output_sizes.end(),
      flat_output.sizes().begin() + 1,
      flat_output.sizes().end());
  torch::jit::push(
      *stack,
      input_physical_views.front().newLogicalFromPhysical(flat_output.view(output_sizes)));
}

} // namespace at
