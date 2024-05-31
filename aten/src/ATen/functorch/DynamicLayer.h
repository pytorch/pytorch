// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/functorch/Macros.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/function_schema.h>
#include <c10/util/Optional.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/functorch/Interpreter.h>
#include <ATen/functorch/VmapInterpreter.h>
#include <ATen/functorch/ADInterpreters.h>
#include <ATen/functorch/FunctionalizeInterpreter.h>

// Forward declared
namespace c10 { struct AutogradMetaInterface; }

namespace at::functorch  {

// This file contains the implementation of functorch's interpreter stack.
// See NOTE: [functorch interpreter stack] first before reading on.
//
// NB: the functorch interpreter stack is also referred to as:
// - the "dynamic layer stack" -- an older name for "interpreter" was
//   "dynamic layer".
// - the "functorch mode stack". You can think of each functorch transform as a
//   "mode" (in the same sense as torch_dispatch mode or torch_function mode),
//   and functorch being an implementation of a "mode stack" where the modes
//   may be arbitrary composed.

// DynamicLayer is basically the same thing as an Interpreter.
// It represents a functorch transform and it holds an Interpreter,
// which contains metadata related to the transform and instructions on
// how to perform the transform.
//
// TODO: we can excise DynamicLayer in favor of Interpreter,
// But I am going to leave it for now as a compatiblity shim to avoid
// needing to refactor a lot of callsites...
struct TORCH_API DynamicLayer {
  explicit DynamicLayer(
      TransformType transform_type,
      int64_t layerId,
      optional<c10::SymInt> batchSize = nullopt,
      optional<RandomnessType> randomness = nullopt,
      optional<bool> prev_grad_mode = nullopt,
      optional<bool> pre_fwd_grad_mode = nullopt,
      optional<bool> functionalize_add_back_views = nullopt);

  TransformType key() const;
  int64_t layerId() const;

  const Interpreter& interpreter() const { return interpreter_; }
  Interpreter& interpreter() { return interpreter_; }

  // Only valid for vmap
  c10::SymInt batchSize() const;
  RandomnessType randomness() const;

 private:
  Interpreter interpreter_;
};

TORCH_API int64_t initAndPushDynamicLayer(
    TransformType transform_type,
    optional<c10::SymInt> batch_size = nullopt,
    optional<RandomnessType> randomness = nullopt,
    optional<bool> prev_grad_mode = nullopt,
    optional<bool> prev_fwd_grad_mode = nullopt,
    optional<bool> functionalize_add_back_views = nullopt);
TORCH_API DynamicLayer popDynamicLayerAndDeleteMetadata();
TORCH_API std::optional<DynamicLayer> maybeCurrentDynamicLayer();
TORCH_API const std::vector<DynamicLayer>& getDynamicLayerStack();
TORCH_API void setDynamicLayerStack(const std::vector<DynamicLayer>& stack);
TORCH_API void setDynamicLayerFrontBackKeysIncluded(bool included);

// NOTE: [Life handles and lexically scoped transforms]
// functorch transforms are lexically scoped.
// Given a level, we store a "life handle" that is a boolean that tells us if the
// transform with that level is active or not.
//
// functorch's TensorWrapper (for grad transforms) stores a life handle.
// If a TensorWrapper escapes from the scope of the transform, then somehow
// it must know it escaped; it can tell by querying the life handle.
TORCH_API const std::shared_ptr<bool>& getLifeHandleForLevel(int64_t level);

// Returns if an operator is in-place. An operator is inplace if:
// 1. The first argument is a Tensor and it is being written to
// 2. The first argument is being returned
// 3. No other arguments are aliased
// Here is an example of an in-place operator:
// add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
TORCH_API bool isInplaceOp(const c10::FunctionSchema& schema);

// Given the indices of unwrapped inputs and the schema, this returns the indices of any outputs that should remain unwrapped
TORCH_API std::optional<size_t> findAliasedOutput(const FunctionSchema& schema, const int64_t immutable_input);

TORCH_API Tensor unwrapIfDead(const Tensor& tensor);
TORCH_API bool isDeadTensorWrapper(const Tensor& tensor);

// Pretty printers
TORCH_API std::ostream& operator<<(std::ostream& os, const DynamicLayer& layer);
TORCH_API std::ostream& operator<<(std::ostream& os, const std::vector<DynamicLayer>& dynamicLayerStack);

// While a functorch transform is active, torch.autograd.function._SingleLevelFunction
// is disabled by default. The following two APIs are APIs for enabling
// it. These are not user-facing APIs. We can delete this in the future, but
// it is useful for debugging when something goes wrong with the
// autograd.Function <> functorch interaction, which uses _SingleLevelFunction,
// because it leads to loud errors if something is incorrect.
TORCH_API void setSingleLevelAutogradFunctionAllowed(bool allowed);
TORCH_API bool getSingleLevelAutogradFunctionAllowed();

// While a functorch grad transform is active, Tensor.requires_grad_() gets
// disabled. These two functions are the mechanism to controlling that.
TORCH_API void setInplaceRequiresGradAllowed(bool allowed);
TORCH_API bool getInplaceRequiresGradAllowed();

TORCH_API DynamicLayer popDynamicLayer();
TORCH_API int64_t pushDynamicLayer(DynamicLayer&& layer);

} // namespace at::functorch
