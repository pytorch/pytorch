// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/functorch/BatchedFallback.h>
#include <ATen/functorch/LegacyVmapTransforms.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/functorch/PlumbingHelper.h>

#include <ATen/Context.h>
#include <ATen/MatrixRef.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/accumulate.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/irange.h>

namespace at {
namespace functorch {

bool kVmapFallbackWarningEnabled = true;

bool isVmapFallbackWarningEnabled() {
  return kVmapFallbackWarningEnabled;
}

void setVmapFallbackWarningEnabled(bool enabled) {
  kVmapFallbackWarningEnabled = enabled;
}

bool kVmapFallbackEnabled = true;

bool isVmapFallbackEnabled() {
  return kVmapFallbackEnabled;
}

void setVmapFallbackEnabled(bool enabled) {
  kVmapFallbackEnabled = enabled;
}

// Given a linear index, return the actual index.
// Example: Given linear_idx = 3, sizes = [5, 2], we would return [1, 0]
static at::SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize>
computeIndex(int64_t linear_idx, IntArrayRef sizes) {
  at::SmallVector<indexing::TensorIndex,kVmapStaticDimVecSize> result;
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

static bool areAllReturnsTensors(const at::FunctionSchema& schema) {
  return std::all_of(
      schema.returns().begin(),
      schema.returns().end(),
      [] (const Argument& arg) { return arg.type() == TensorType::get(); });
}

static bool areAnyArgumentsTensorList(const at::FunctionSchema& schema) {
  return std::any_of(
      schema.arguments().begin(),
      schema.arguments().end(),
      [] (const Argument& arg) {
        return arg.type()->isSubtypeOf(ListType::ofTensors()) ||
          arg.type()->isSubtypeOf(ListType::ofOptionalTensors());
      });
}

static void warnFallback(const c10::FunctionSchema& schema, bool is_inplace, bool is_nested=false) {
  TORCH_CHECK(isVmapFallbackEnabled(),
      schema.operator_name(), " hit the vmap fallback which is currently disabled");
  if (!isVmapFallbackWarningEnabled()) {
    return;
  }
  TORCH_WARN("There is a performance drop because we have not yet implemented ",
             "the ", (is_nested ? "nested " : "") , "batching rule for ",
             schema.operator_name(), ". Please file us an issue on GitHub so that ",
             "we can prioritize its implementation.");
}

// The general flow of the algorithm is as follows.
// - First, we figure out which arguments are BatchedTensors and save them
//   to a vector. We also store a vector of which index of the arguments list
//   each BatchedTensor appears in. This will be useful for bookkeeping later.
// - Next, we apply the MultiBatchVmapTransform to all of the BatchedTensors.
//   This returns a vector of VmapPhysicalView that hold tensors that contain
//   all of the collective batch dimensions at the front of the tensors.
// - Then, we attempt to call `op` once per slice of the inputs. To do this,
//   we repeatedly we slice the input arguments (if they are BatchedTensors),
//   put the sliced (or a not-sliced) version of the input onto the stack, invoke
//   the operator, and then pop the results off the stack.
static void batchedTensorInplaceForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  warnFallback(schema, /*in_place*/true);

  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  // `self` is the Tensor being modified in-place
  Tensor self = arguments[0].toTensor();
  const auto* self_impl = maybeGetBatchedImpl(self);
  std::bitset<kVmapMaxTensorDims> self_vmap_levels;
  if (self_impl) {
    self_vmap_levels = createVmapLevelsBitset(self_impl->level());
  }

  // Figure out which arguments are BatchedTensor. Save them to a vector.
  // For each BatchedTensor, also record what position of `arguments` they came from.
  at::SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(0, arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.defined()) {
      continue;
    }
    const auto* batched = maybeGetBatchedImpl(tensor);
    if (!batched) {
      continue;
    }

    // NOTE: [vmap-incompatible in-place operations]
    // In-place operations on `self` are not possible if there exists some vmap
    // level `l` such that `self` is not being vmapped on that level but another
    // argument is. For example, let B0 be a batch dim inside vmap and consider
    // vmap(Tensor.add_, in_dims=(None, 0))(torch.ones(3), torch.ones(B0, 3))
    // - self is torch.ones(3) and does not participate in this vmap
    // - other is BatchedTensor(torch.ones(B0, 3))
    // There's no way to do self.add_(other) because `other` has more elements
    // elements than `self` due to being vmapped over.
    //
    // In the vmap fallback, we should error out when we detect this.
    auto other_vmap_levels = createVmapLevelsBitset(batched->level());
    if (self_vmap_levels != (self_vmap_levels | other_vmap_levels)) {
      // Find one vmap level to complain about
      auto additional_bdims = (self_vmap_levels | other_vmap_levels) ^ self_vmap_levels;
      auto offending_level = llvm::findLastSet(additional_bdims.to_ulong());
      // The following prints out "vmap: aten::add_(tensor, ...) is not possible",
      // but it would be better to print out "tensor.add_(...) is not possible".
      // Afaict there's no official way to get the add_ and there is no way to
      // tell if an operator has method or function variants.
      TORCH_CHECK(false,
        "vmap: ", schema.name(), "(self, *extra_args) is not possible because ",
        "there exists a Tensor `other` in extra_args that has more elements ",
        "than `self`. This happened due to `other` being vmapped over but ",
        "`self` not being vmapped over at level ", offending_level, ". ",
        "Please try to use out-of-place operators instead of ", schema.name(), ". ",
        "If said operator is being called inside the PyTorch framework, ",
        "please file a bug report instead.");
    }
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(idx);
  }
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // MultiBatchVmapTransform the BatchedTensor arguments. This returns
  // VmapPhysicalViews that contain all of the batch dimensions.
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);

  // Compute the total number of batches
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  auto first_physical_view_sizes = input_physical_views.front().tensor().sizes();
  auto batch_sizes = ArrayRef<int64_t>(
      first_physical_view_sizes.begin(), first_physical_view_sizes.begin() + num_batch_dims);
  const auto num_batches = c10::multiply_integers(batch_sizes);
  // Without a shape-checking API, we're unable to compute the correct shape of
  // the output so we just error out.
  TORCH_CHECK(num_batches > 0,
      "Batching rule not implemented for ", schema.operator_name(), ". ",
      "The fallback path does not support vmap over dims of size 0.");

  // Strategy: For each batch, we are going to push slices (where applicable)
  // of the arguments onto `stack`, and call `op`.
  for (int64_t linear_idx = 0; linear_idx < num_batches; ++linear_idx) {
    auto index = computeIndex(linear_idx, batch_sizes);
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    auto input_physical_views_iter = input_physical_views.begin();
    for (const auto arg_idx : c10::irange(0, num_arguments)) {
      // We assume that torch::jit::Stack is backed by vector<IValue> for
      // simplicity. When that is not the case, this code should be updated.
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || (int64_t)arg_idx != *batched_tensor_inputs_pos_iter) {
        // argument isn't a BatchedTensor
        torch::jit::push(stack, argument);
        continue;
      }
      // argument is a BatchedTensor
      TORCH_INTERNAL_ASSERT(input_physical_views_iter != input_physical_views.end());
      const auto& physical_view_for_argument = *input_physical_views_iter;
      auto thing = physical_view_for_argument.tensor().index(index);
      torch::jit::push(stack, thing);
      batched_tensor_inputs_pos_iter++;
      input_physical_views_iter++;
    }

    op.callBoxed(stack);
    torch::jit::drop(stack, 1);
  }

  // Return the tensor that was written to in-place
  torch::jit::drop(stack, num_arguments);
  torch::jit::push(stack, self);
}

static Tensor safeStack(TensorList tensors) {
  auto is_defined = [](const Tensor& t) { return t.defined(); };
  if (std::all_of(tensors.begin(), tensors.end(), is_defined)) {
    return at::stack(tensors);
  }
  // NOTE [vmap through backward and undefined grad]
  // While vmapping through backward functions (to compute batched grad), it
  // is possible for the backward function to return an undefined grad for some
  // grad_input for each example. In that case, we return an undefined grad.
  //
  // It is theoretically posssible for *some* of the examples to produce an
  // undefined grad (a kernel could peek at the gradient values and return an
  // undefined tensor if it determines the gradient is full of zeros). We
  // could handle this by treating the undefined grad as a zero-filled tensor
  // of the correct shape while stacking the tensors together. However I expect
  // this to happen very rarely (I have not been able to find an example in our
  // codebase) so we just error out in this case.
  if (std::none_of(tensors.begin(), tensors.end(), is_defined)) {
    return Tensor();
  }
  TORCH_CHECK(false,
      "vmap: slow fallback received a mix of undefined and defined tensors ",
      "as the result of an operation. This is not supported, please file us ",
      "an issue on github.");
}

// TODO: Consider rewriting the following to look like:
// https://gist.github.com/zou3519/7b7c6a4a258d580f62d1d969851be6b1<Paste>

// The general flow of the algorithm is as follows.
// - First, we figure out which arguments are BatchedTensors and save them
//   to a vector. We also store a vector of which index of the arguments list
//   each BatchedTensor appears in. This will be useful for bookkeeping later.
// - Next, we apply the MultiBatchVmapTransform to all of the BatchedTensors.
//   This returns a vector of VmapPhysicalView that hold tensors that contain
//   all of the collective batch dimensions at the front of the tensors.
// - Then, we attempt to call `op` once per slice of the inputs. To do this,
//   we repeatedly we slice the input arguments (if they are BatchedTensors),
//   put the sliced (or a not-sliced) version of the input onto the stack, invoke
//   the operator, and then pop the results off the stack.
// - Each result obtained from the previous step is a slice of the total result,
//   so we stack those tensors together to form the final result.
void batchedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);

  TORCH_CHECK(areAllReturnsTensors(schema) && !areAnyArgumentsTensorList(schema),
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "We could not generate a fallback.");

  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    op.callBoxed(stack);
    return;
  }

  if (isInplaceOp(schema)) {
    batchedTensorInplaceForLoopFallback(op, stack);
    return;
  }
  TORCH_CHECK(!schema.is_mutable() && !schema.hasAnyAliasInfo(),
              "Batching rule not implemented for ", schema.operator_name(), "; ",
              "the fallback path doesn't work on out= or view ops.");
  TORCH_CHECK(num_returns >= 1,
              "Batching rule not implemented for ", schema.operator_name(), ". ",
              "The fallback path does not support operations with no returns.");
  warnFallback(schema, /*in_place*/false);

  const auto arguments_begin = stack->size() - num_arguments;

  // Figure out which arguments are BatchedTensor. Save them to a vector.
  // For each BatchedTensor, also record what position of `arguments` they came from.
  at::SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(0, arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.defined()) {
      continue;
    }
    const auto* batched = maybeGetBatchedImpl(tensor);
    if (!batched) {
      continue;
    }
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(idx);
  }
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  // MultiBatchVmapTransform the BatchedTensor arguments. This returns
  // VmapPhysicalViews that contain all of the batch dimensions.
  const auto input_physical_views = MultiBatchVmapTransform::logicalToPhysical(
      batched_tensor_inputs);

  // Compute the total number of batches
  auto num_batch_dims = input_physical_views.front().numBatchDims();
  auto some_sizes = input_physical_views.front().tensor().sizes();
  auto batch_sizes = ArrayRef<int64_t>(some_sizes.begin(), some_sizes.begin() + num_batch_dims);
  const auto num_batches = c10::multiply_integers(batch_sizes);
  // Without a shape-checking API, we're unable to compute the correct shape of
  // the output so we just error out.
  TORCH_CHECK(num_batches > 0,
      "Batching rule not implemented for ", schema.operator_name(), ". ",
      "The fallback path does not support vmap over dims of size 0.");

  // Strategy: For each batch, we are going to push slices (where applicable)
  // of the arguments onto `stack`, call `op`, and store the result in
  // `output_shards`.
  //
  // NOTE: [Output shards layout]
  // Assume that the operator has three outputs: a, b, c.
  // The layout of output_shards is as follows:
  // [ a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3]
  // This is so that we can call at::stack([a0...a3]), at::stack([b0...b3])
  // more easily in the next step.
  std::vector<Tensor> output_shards(num_batches * num_returns);

  for (int64_t linear_idx = 0; linear_idx < num_batches; ++linear_idx) {
    auto index = computeIndex(linear_idx, batch_sizes);
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    auto input_physical_views_iter = input_physical_views.begin();
    for (const auto arg_idx : c10::irange(0, num_arguments)) {
      // We assume that torch::jit::Stack is backed by vector<IValue> for
      // simplicity. When that is not the case, this code should be updated.
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || (int64_t)arg_idx != *batched_tensor_inputs_pos_iter) {
        // argument isn't a BatchedTensor
        torch::jit::push(stack, argument);
        continue;
      }
      // argument is a BatchedTensor
      TORCH_INTERNAL_ASSERT(input_physical_views_iter != input_physical_views.end());
      const auto& physical_view_for_argument = *input_physical_views_iter;
      c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
      torch::jit::push(stack, physical_view_for_argument.tensor().index(index));
      batched_tensor_inputs_pos_iter++;
      input_physical_views_iter++;
    }

    // std::cout << "[Fallback]: ";
    // at::dump_tensor((*stack)[stack->size() - 1].toTensor());
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    op.callBoxed(stack);

    // Store the result into `output_shards`. See NOTE: [Output shards layout]
    // to learn about the details of how we store the shards.
    const auto returns = torch::jit::last(stack, num_returns);
    for (const auto  return_idx : c10::irange(0, returns.size())) {
      output_shards[num_batches * return_idx + linear_idx] = returns[return_idx].toTensor();
    }
    torch::jit::drop(stack, num_returns);
  }

  // For each output Tensor, stack the shards of the tensor together to form a return
  torch::jit::drop(stack, num_arguments);
  auto output_shards_chunks = MatrixRef<Tensor>(output_shards, num_batches);
  for (const auto return_idx : c10::irange(0, num_returns)) {
    auto shards = output_shards_chunks[return_idx];
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    auto flat_output = safeStack(shards);
    // See NOTE [vmap through backward and undefined grad]
    if (!flat_output.defined()) {
      torch::jit::push(stack, flat_output);
      continue;
    }
    VmapDimVector output_sizes(batch_sizes);
    output_sizes.insert(
        output_sizes.end(),
        flat_output.sizes().begin() + 1,
        flat_output.sizes().end());
    torch::jit::push(
        stack,
        input_physical_views.front().getPhysicalToLogicalMap().apply(flat_output.view(output_sizes)));
  }
}

void batchedNestedTensorForLoopFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();
  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);

  TORCH_CHECK(areAllReturnsTensors(schema) && !areAnyArgumentsTensorList(schema),
              "Nested batching rule not implemented for ", schema.operator_name(), ". ",
              "We could not generate a fallback.");

  if (std::none_of(arguments.begin(), arguments.end(), ivalueParticipatesInCurrentLevel)) {
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::FuncTorchBatched);
    c10::impl::ExcludeDispatchKeyGuard nt_guard(DispatchKey::BatchedNestedTensor);
    op.callBoxed(stack);
    return;
  }

  if (isInplaceOp(schema)) {
    TORCH_INTERNAL_ASSERT(false, "vmap fallback not supported for in-place ops on nested tensors");
    return;
  }
  TORCH_CHECK(!schema.is_mutable() && !schema.hasAnyAliasInfo(),
              "Nested batching rule not implemented for ", schema.operator_name(), "; ",
              "the fallback path doesn't work on out= or view ops.");
  TORCH_CHECK(num_returns >= 1,
              "Nested batching rule not implemented for ", schema.operator_name(), ". ",
              "The fallback path does not support operations with no returns.");
  warnFallback(schema, /*in_place*/false, /*is_nested*/true);

  const auto arguments_begin = stack->size() - num_arguments;

  // Figure out which arguments are BatchedTensor. Save them to a vector.
  // For each BatchedTensor, also record what position of `arguments` they came from.
  at::SmallVector<Tensor,kVmapTransformStaticInputSize> batched_tensor_inputs;
  VmapDimVector batched_tensor_inputs_position;
  for (const auto idx : c10::irange(0, arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (!ivalue.isTensor()) {
      continue;
    }
    const auto& tensor = ivalue.toTensor();
    if (!tensor.defined()) {
      continue;
    }
    const auto* batched = maybeGetBatchedImpl(tensor);
    if (!batched) {
      continue;
    }
    batched_tensor_inputs.push_back(tensor);
    batched_tensor_inputs_position.push_back(idx);
  }
  TORCH_INTERNAL_ASSERT(!batched_tensor_inputs.empty());

  std::vector<std::vector<Tensor>> unbound;
  for (auto iter = batched_tensor_inputs.begin(); iter != batched_tensor_inputs.end(); ++iter) {
    auto *batched_impl = maybeGetBatchedImpl(*iter);
    TORCH_INTERNAL_ASSERT(batched_impl->value().is_nested() || batched_impl->bdim() == 0,
        "Fallback not supported for mixed nested / non-nested arguments without bdim=0");
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    auto this_unbound = batched_impl->value().unbind();
    if (unbound.size() > 0) {
      TORCH_INTERNAL_ASSERT(unbound.front().size() == this_unbound.size(),
          "Fallback not supported for differently-sized nested arguments");
    }
    unbound.push_back(this_unbound);
  }

  const auto num_components = unbound.front().size();
  std::vector<Tensor> output_shards(num_components * num_returns);
  for (const auto component_idx : c10::irange(0, num_components)) {
    auto batched_idx = 0;
    auto batched_tensor_inputs_pos_iter = batched_tensor_inputs_position.begin();
    for (const auto arg_idx : c10::irange(0, num_arguments)) {
      // We assume that torch::jit::Stack is backed by vector<IValue> for
      // simplicity. When that is not the case, this code should be updated.
      const auto& argument = (*stack)[arguments_begin + arg_idx];
      if (batched_tensor_inputs_pos_iter == batched_tensor_inputs_position.end()
          || (int64_t)arg_idx != *batched_tensor_inputs_pos_iter) {
        // argument isn't a BatchedTensor
        torch::jit::push(stack, argument);
        continue;
      }
      // argument is a BatchedTensor
      torch::jit::push(stack, unbound[batched_idx][component_idx]);
      ++batched_idx;
      ++batched_tensor_inputs_pos_iter;
    }

    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    op.callBoxed(stack);

    // Store the result into `output_shards`. See NOTE: [Output shards layout]
    // to learn about the details of how we store the shards.
    const auto returns = torch::jit::last(stack, num_returns);
    for (const auto return_idx : c10::irange(0, returns.size())) {
      output_shards[num_components * return_idx + component_idx] = returns[return_idx].toTensor();
    }
    torch::jit::drop(stack, num_returns);
  }

  // For each output Tensor, stack the shards of the tensor together to form a nested return
  // TODO: Determine when the output needs to be nested and when it can be non-nested?
  torch::jit::drop(stack, num_arguments);
  auto output_shards_chunks = MatrixRef<Tensor>(output_shards, num_components);
  for (const auto return_idx : c10::irange(0, num_returns)) {
    auto shards = output_shards_chunks[return_idx];
    c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::BatchedNestedTensor);
    auto out_nt = at::_nested_tensor_from_tensor_list(shards);
    // NB: NTs only support batching over dim 0
    torch::jit::push(stack, makeBatched(out_nt, 0, maybeCurrentDynamicLayer()->layerId()));
  }
}

void vmapErrorFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  TORCH_CHECK(false, "Error: ", op.operator_name(), " requires special handling, and does not yet have a batching rule. Feel free to file a github issue!");
}

}
} // namespace at
