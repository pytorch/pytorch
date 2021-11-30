// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/BatchedTensorImpl.h>

#include <torch/library.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/variable.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <c10/util/irange.h>

namespace at {
namespace functorch {

// Initial autograd layer, because autograd is always "on"
// thread_local std::vector<DynamicLayer> dynamicLayerStack = { DynamicLayer(DispatchKey::Autograd, 1) };

using DynmetaData = std::unordered_map<int64_t, std::shared_ptr<bool>>;
DynmetaData kDynMetaDataSingleton;

static DynmetaData& getGlobalDynmetaData() {
  return kDynMetaDataSingleton;
}

class DynamicLayerStackHolder : public c10::DebugInfoBase {
 public:
  DynamicLayerStackHolder() {}
  virtual ~DynamicLayerStackHolder() {}

  std::vector<DynamicLayer> dynamicLayerStack = { DynamicLayer(DispatchKey::Autograd, 1, nullopt, true) };
};

thread_local std::shared_ptr<DynamicLayerStackHolder> kDynamicLayerStack;

static std::vector<DynamicLayer>& dynamicLayerStackAccessor() {
  if (kDynamicLayerStack != nullptr) {
    // TODO: can figure out how to memoize this. std::call_once with thread_local?
    return kDynamicLayerStack->dynamicLayerStack;
  }
  if (ThreadLocalDebugInfo::current() != nullptr) {
    // TODO: this is going to break if someone else uses PRODUCER_INFO
    kDynamicLayerStack = std::static_pointer_cast<DynamicLayerStackHolder>(
        ThreadLocalDebugInfo::_peek(c10::DebugInfoKind::PRODUCER_INFO));
    TORCH_INTERNAL_ASSERT(kDynamicLayerStack != nullptr);
    return kDynamicLayerStack->dynamicLayerStack;
  }
  kDynamicLayerStack = std::make_shared<DynamicLayerStackHolder>();
  c10::ThreadLocalDebugInfo::_push(
      // TODO: this isn't a PRODUCER_INFO, but there's nothing else we can use
      c10::DebugInfoKind::PRODUCER_INFO,
      kDynamicLayerStack);
  TORCH_INTERNAL_ASSERT(kDynamicLayerStack != nullptr);
  return kDynamicLayerStack->dynamicLayerStack;
}

std::shared_ptr<bool> getLifeHandleForLevel(int64_t level) {
  auto it = getGlobalDynmetaData().find(level);
  TORCH_INTERNAL_ASSERT(it != kDynMetaDataSingleton.end(), "level should be alive");
  return it->second;
}

optional<DynamicLayer> maybeCurrentDynamicLayer() {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  // NB: Exception for regular autograd, maybe tweak this
  if (dynamicLayerStack.size() <= 1) {
    return {};
  }
  return dynamicLayerStack.back();
}

const std::vector<DynamicLayer>& getDynamicLayerStack() {
  return dynamicLayerStackAccessor();
}

void setDynamicLayerStack(const std::vector<DynamicLayer>& stack) {
  dynamicLayerStackAccessor() = stack;
}

bool areTransformsActive() {
  const auto& data = getGlobalDynmetaData();
  return !data.empty();
}

static DynamicLayer popDynamicLayer() {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
  auto result = dynamicLayerStack.back();
  TORCH_INTERNAL_ASSERT(result.key() != DispatchKey::Undefined);
  dynamicLayerStack.pop_back();

  if (dynamicLayerStack.size() == 0) {
    // std::cout << "DynamicLayer off" << std::endl;
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerFrontModeKey, false);
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerBackModeKey, false);
  }

  return result;
}

static int64_t pushDynamicLayer(DynamicLayer&& dynamic_layer) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  int64_t layerId = 1 + dynamicLayerStack.size();
  TORCH_INTERNAL_ASSERT(layerId == dynamic_layer.layerId());
  dynamicLayerStack.emplace_back(dynamic_layer);

  if (layerId == 2) {
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerFrontModeKey, true);
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerBackModeKey, true);
  }

  return layerId;
}

static int64_t pushDynamicLayer(
    DispatchKey key,
    optional<int64_t> batch_size = nullopt,
    optional<bool> prev_grad_mode = nullopt) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  TORCH_INTERNAL_ASSERT(key != DispatchKey::Undefined);
  TORCH_INTERNAL_ASSERT(key != DispatchKey::Batched);

  auto layerId = 1 + dynamicLayerStack.size();
  dynamicLayerStack.emplace_back(key, layerId, batch_size, prev_grad_mode);

  if (layerId == 2) {
    // std::cout << "DynamicLayer on" << std::endl;
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerFrontModeKey, true);
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerBackModeKey, true);
  }

  return layerId;
}

int64_t initAndPushDynamicLayer(
    DispatchKey key,
    optional<int64_t> batch_size,
    optional<bool> prev_grad_mode) {
  auto layerId = pushDynamicLayer(key, batch_size, prev_grad_mode);
  auto& data = getGlobalDynmetaData();
  TORCH_INTERNAL_ASSERT(data.find(layerId) == data.end());
  if (key == DispatchKey::Autograd) {
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }
  data[layerId] = std::make_shared<bool>(true);
  return layerId;
}

DynamicLayer popDynamicLayerAndDeleteMetadata() {
  auto result = popDynamicLayer();
  auto level = result.layerId();

  // TODO: is this lock safe? No one else should be writing to the same bucket
  // if (c10::show_dispatch_trace_enabled()) {
  //   std::cout << "deleting metadata" << std::endl;
  // }
  auto& data = getGlobalDynmetaData();
  auto it = data.find(level);
  if (it == data.end()) {
    return result;
  }
  // if (c10::show_dispatch_trace_enabled()) {
  //   std::cout << "deleted metadata for level " << level << std::endl;
  // }
  // invalidate the thing
  *(it->second) = false;
  data.erase(level);
  return result;
}

static Tensor materializeGradWrappers(const Tensor& tensor, const std::vector<DynamicLayer>& dynlayerStack) {
  if (!tensor.defined()) {
    return tensor;
  }
  // TODO: First entry in the stack is a default autograd key.
  // We should clean up the logic
  if (dynlayerStack.size() <= 1) {
    return tensor;
  }
  if (dynlayerStack.back().key() != DispatchKey::Autograd) {
    return tensor;
  }
  auto cur_level = dynlayerStack.back().layerId();
  auto* wrapper = maybeGetTensorWrapper(tensor);
  if (!wrapper) {
    return makeTensorWrapper(tensor, cur_level);
  }
  TORCH_INTERNAL_ASSERT(wrapper->level().value() <= cur_level, "escaped?");
  if (wrapper->level().value() == cur_level) {
    TORCH_INTERNAL_ASSERT(tensor.defined());
    return tensor;
  }
  return makeTensorWrapper(tensor, cur_level);
}

static Tensor unwrapIfDead(const Tensor& tensor) {
  auto* wrapped = maybeGetTensorWrapper(tensor);
  if (!wrapped) {
    return tensor;
  }
  if (wrapped->is_alive()) {
    return tensor;
  }
  return wrapped->value();
}

void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
    std::function<Tensor(const Tensor&)> func) {
  TORCH_INTERNAL_ASSERT(begin >= 0);
  TORCH_INTERNAL_ASSERT(end >= 0);
  TORCH_INTERNAL_ASSERT(begin <= end);
  for (int64_t idx = begin; idx < end; idx++) {
    auto ivalue = args[idx];
    // Tensor?[] translates to a c10::List<IValue> so we need to peek inside List
    if (ivalue.isList()) {
      bool modified = false;
      // TODO: might be more efficient if we scan first then not copy? Depends.
      auto list = ivalue.toList().copy();
      for (const auto list_idx : c10::irange(0, list.size())) {
        const auto& elt = list.get(list_idx);
        if (elt.isTensor()) {
          list.set(list_idx, func(elt.toTensor()));
          modified = true;
        }
      }
      if (modified) {
        args[idx] = list;
      }
      continue;
    }
    if (ivalue.isTensorList()) {
      auto list = ivalue.toTensorList();
      for (const auto list_idx : c10::irange(0, list.size())) {
        list[list_idx] = func(list[list_idx]);
      }
      args[idx] = list;
    }
    TORCH_INTERNAL_ASSERT(!ivalue.isGenericDict(), "No operators can accept GenericDict");
    if (!ivalue.isTensor()) {
      continue;
    }
    Tensor value = ivalue.toTensor();
    Tensor replacement = func(value);
    args[idx] = std::move(replacement);
    // sanity checks
    if (ivalue.toTensor().defined()) {
      TORCH_INTERNAL_ASSERT(args[idx].toTensor().defined());
    }
  }
}

static bool allTensors(
    ArrayRef<IValue> args,
    std::function<bool(const Tensor&)> pred) {
  for (const auto& ivalue : args) {
    // Tensor?[] translates to a c10::List<IValue> so we need to peek inside List
    if (ivalue.isList()) {
      for (const auto& elt : ivalue.toListRef()) {
        if (elt.isTensor() && !pred(elt.toTensor())) {
            return false;
        }
      }
      continue;
    }
    if (ivalue.isTensorList()) {
      for (const auto& elt : ivalue.toTensorList()) {
        if (!pred(elt)) {
          return false;
        }
      }
      continue;
    }
    TORCH_INTERNAL_ASSERT(!ivalue.isGenericDict(), "No operators can accept GenericDict");
    if (!ivalue.isTensor()) {
      continue;
    }
    if (!pred(ivalue.toTensor())) {
      return false;
    }
  }
  return true;
}

static bool anyTensors(
    ArrayRef<IValue> args,
    std::function<bool(const Tensor&)> pred) {
  // Demorgan's law
  return !allTensors(args, [&](const Tensor& self) { return !pred(self); });
}

constexpr DispatchKeySet all_dynlayer_keyset = DispatchKeySet({
  kDynamicLayerFrontModeKey,
  kDynamicLayerBackModeKey,
  kGradWrapperKey,
  // DispatchKey::Batched,
  kBatchedKey,
  DispatchKey::ADInplaceOrView
}) | autograd_dispatch_keyset;

static void sanityCheckStack(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(),
      [](const Tensor& tensor) {

        auto* wrapper = maybeGetTensorWrapper(tensor);
        TORCH_INTERNAL_ASSERT(wrapper == nullptr);
        auto* batched = maybeGetBatchedImpl(tensor);
        TORCH_INTERNAL_ASSERT(batched == nullptr);
        return tensor;
      });
}

static bool batchedAtCurrentLevel(const Tensor& tensor) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  auto layer = dynamicLayerStack.back();
  auto level = layer.layerId();

  auto* batched = maybeGetBatchedImpl(tensor);
  if (!batched) {
    return false;
  }
  auto batched_at_level = batched->level();
  return batched_at_level == level;
}

bool isInplaceOp(const FunctionSchema& schema) {
  if (!schema.is_mutable() || schema.returns().size() != 1) {
    return false;
  }
  // Check that the first argument is being written to
  const auto& first_arg_alias_info = schema.arguments().begin()->alias_info();
  if (!first_arg_alias_info || !first_arg_alias_info->isWrite()) {
    return false;
  }
  // Check that none of the other args are being aliased
  for (auto it = schema.arguments().begin() + 1; it != schema.arguments().end(); ++it) {
    const auto& alias_info = it->alias_info();
    if (alias_info) {
      return false;
    }
  }
  // Check that the first tensor is being returned (i.e., output has a (a!))
  const auto& return_alias_info = schema.returns()[0].alias_info();
  return return_alias_info && return_alias_info->isWrite();
}

static void checkForInvalidMutationOnCaptures(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    const std::vector<DynamicLayer>& dynamicLayerStack) {
  if (dynamicLayerStack.back().key() != DispatchKey::Autograd) {
    return;
  }
  // TODO: First entry in the stack is a default autograd key.
  // We should clean up the logic
  if (dynamicLayerStack.size() <= 1) {
    return;
  }
  if (!isInplaceOp(op.schema())) {
    return;
  }
  auto args = torch::jit::last(stack, op.schema().arguments().size());
  auto mutated_arg = unwrapIfDead(args[0].toTensor());
  auto cur_level = dynamicLayerStack.back().layerId();
  auto* wrapper = maybeGetTensorWrapper(mutated_arg);
  if (wrapper && wrapper->level().has_value() && wrapper->level().value() == cur_level) {
    return;
  }
  TORCH_CHECK(false,
      "During a grad (vjp, jvp, grad, etc) transform, the function provided ",
      "attempted to call in-place operation (", op.schema().operator_name(), ") ",
      "that would mutate a captured Tensor. This is not supported; please rewrite ",
      "the function being transformed to explicitly accept the mutated Tensor(s) ",
      "as inputs.");
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
  if (c10::show_dispatch_trace_enabled()) {
    std::cout << "DLS size: " << dynamicLayerStack.size() << std::endl;
  }
#endif
  if (dynamicLayerStack.size() == 0) {
    sanityCheckStack(op, stack);
    c10::impl::ExcludeDispatchKeyGuard guard(all_dynlayer_keyset);
    op.callBoxed(stack);
    return;
  }

  // if is a grad transform, and the operation is in-place, and the mutated
  // argument is not currently wrapped in a TensorWrapper, then we need to
  // error out otherwise the result is silently incorrect
  checkForInvalidMutationOnCaptures(op, stack, dynamicLayerStack);

  // Unwrap dead GradWrappers, materialize live ones
  auto maybeTransformGradWrappers = [](const Tensor& tensor) {
    auto result = unwrapIfDead(tensor);
    return materializeGradWrappers(result, getDynamicLayerStack());
  };
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(), maybeTransformGradWrappers);

  auto layer = dynamicLayerStack.back();

  DispatchKeySet exclude = all_dynlayer_keyset;
  exclude = exclude.remove(kDynamicLayerBackModeKey);
  DispatchKeySet include;
  if (layer.key() == DispatchKey::Autograd) {
    exclude = exclude - autograd_dispatch_keyset;
    exclude = exclude.remove(DispatchKey::ADInplaceOrView);
  } else if (layer.key() == kBatchedKey) {
    // Only enable dispatch on kBatchedKey if there are tensors batched
    // at the current level.
    const auto args = torch::jit::last(stack, op.schema().arguments().size());
    if (anyTensors(args, batchedAtCurrentLevel)) {
      exclude = exclude.remove(kBatchedKey);
    }
    include = include.add(kVmapModeKey);
  } else {
    TORCH_INTERNAL_ASSERT(false);
  }
  c10::impl::ExcludeDispatchKeyGuard exclude_guard(exclude);
  c10::impl::IncludeDispatchKeyGuard include_guard(include);

  // Re-dispatch
  op.callBoxed(stack);
}

struct WithoutTop {
  WithoutTop(): layer_(popDynamicLayer()) {
  }
  ~WithoutTop() {
    pushDynamicLayer(std::move(layer_));
  }

  DynamicLayer layer_;
};

struct SaveLocalDispatchKeySet {
 public:
  SaveLocalDispatchKeySet() :
    saved_keyset_(c10::impl::tls_local_dispatch_key_set()) {}
  ~SaveLocalDispatchKeySet() {
    c10::impl::_force_tls_local_dispatch_key_set(saved_keyset_);
  }

 private:
  c10::impl::LocalDispatchKeySet saved_keyset_;
};

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto cur_level = getDynamicLayerStack().back().layerId();
  auto cur_key = getDynamicLayerStack().back().key();

  optional<bool> prev_grad_mode = getDynamicLayerStack().back().prevGradMode();
  if (cur_key == DispatchKey::Autograd) {
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }

  auto unwrap = [&](const Tensor& tensor) {
    if (!tensor.defined()) {
      return tensor;
    }
    auto* maybe_tensor_wrapper = maybeGetTensorWrapper(tensor);
    if (!maybe_tensor_wrapper) {
      return tensor;
    }
    auto tensor_wrapper_level = maybe_tensor_wrapper->level().value();
    TORCH_INTERNAL_ASSERT(tensor_wrapper_level <= cur_level);
    if (tensor_wrapper_level == cur_level) {
      return maybe_tensor_wrapper->value();
    }
    return tensor;
  };
  auto wrap = [&](const Tensor& tensor) {
    if (!tensor.defined()) {
      return tensor;
    }
    if (cur_level == 1) {
      return tensor;
    }
    // if (c10::show_dispatch_trace_enabled()) {
    //   std::cout << "wrap " << cur_level << std::endl;
    // }
    return makeTensorWrapper(tensor, cur_level);
  };

  // TODO: we only need to do the following (marked with !) on in-place functions
  // that modify sizes or strides. There aren't many of them.
  // If autograd dispatch key:
  // 1. (!) Put a copy of all of the args onto the stack
  // 2. Unwrap all the args in the copy set
  // 3. Call the operator
  // 4. Wrap the output
  // 5. (!) refreshMetadata for all the args in the original set
  // 6. (!) Pop those args off.

  // Step 1 & 2
  if (cur_key == DispatchKey::Autograd) {
    auto args_size = op.schema().arguments().size();
    // Step 1
    auto front = stack->size() - args_size;
    for (const auto arg_idx : c10::irange(0, args_size)) {
      stack->push_back((*stack)[front + arg_idx]);
    }
    // Step 2
    foreachTensorInplace(*stack, stack->size() - args_size, stack->size(), unwrap);
  }

  // pop the top layer. Put it back on dtor.
  WithoutTop guard;

  // "reset exclude set"
  // TODO: Still a problem with composabiilty and AutoNonVariableTypeGuard.
  // Users cannot do torch.no_grad otherwise there will be problems.
  SaveLocalDispatchKeySet save_guard;
  auto keyset = c10::impl::PODLocalDispatchKeySet();
  c10::impl::_force_tls_local_dispatch_key_set(keyset);
  c10::impl::tls_set_dispatch_key_included(kDynamicLayerFrontModeKey, true);
  c10::impl::tls_set_dispatch_key_included(kDynamicLayerBackModeKey, true);

  // Re-dispatch
  if (cur_key == DispatchKey::Autograd && *prev_grad_mode == false) {
    // See NOTE [grad and vjp interaction with no_grad]
    c10::AutoGradMode guard(*prev_grad_mode);
    op.callBoxed(stack);
  } else {
    op.callBoxed(stack);
  }

  // Step 4, 5, 6
  if (cur_key == DispatchKey::Autograd) {
    // Step 4
    auto ret_size = op.schema().returns().size();
    foreachTensorInplace(*stack, stack->size() - ret_size, stack->size(), wrap);

    // Step 5
    auto args_size = op.schema().arguments().size();
    auto args_front = stack->size() - args_size - ret_size;
    for (const auto arg_idx : c10::irange(0, args_size)) {
      auto& ivalue = (*stack)[args_front + arg_idx];
      if (!ivalue.isTensor()) {
        continue;
      }
      auto maybe_tensor_wrapper = maybeGetTensorWrapper(ivalue.toTensor());
      if (!maybe_tensor_wrapper) {
        continue;
      }
      maybe_tensor_wrapper->refreshMetadata();
    }

    // Step 6
    stack->erase(stack->end() - (args_size + ret_size), stack->end() - ret_size);
  }
}

TORCH_LIBRARY_IMPL(_, FT_DYNAMIC_LAYER_FRONT_MODE_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallback>());
}

TORCH_LIBRARY_IMPL(_, FT_DYNAMIC_LAYER_BACK_MODE_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerBackFallback>());
}

// TORCH_LIBRARY_IMPL(aten, DynamicLayerFront, m) {
//   m.impl("_unwrap_for_grad", native::_unwrap_for_grad);
//   m.impl("dump_tensor", native::dump_tensor);
//   m.impl("dlevel", native::dlevel);
// }

}
} // namespace at
