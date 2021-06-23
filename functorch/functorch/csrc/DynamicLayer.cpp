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

  std::vector<DynamicLayer> dynamicLayerStack = { DynamicLayer(DispatchKey::Autograd, 1) };
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

static int64_t pushDynamicLayer(DispatchKey key, optional<int64_t> batch_size = nullopt) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  TORCH_INTERNAL_ASSERT(key != DispatchKey::Undefined);
  TORCH_INTERNAL_ASSERT(key != DispatchKey::Batched);
  auto layerId = 1 + dynamicLayerStack.size();
  dynamicLayerStack.emplace_back(key, layerId, batch_size);

  if (layerId == 2) {
    // std::cout << "DynamicLayer on" << std::endl;
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerFrontModeKey, true);
    c10::impl::tls_set_dispatch_key_included(kDynamicLayerBackModeKey, true);
  }

  return layerId;
}

int64_t initAndPushDynamicLayer(DispatchKey key, optional<int64_t> batch_size) {
  auto layerId = pushDynamicLayer(key, batch_size);
  auto& data = getGlobalDynmetaData();
  TORCH_INTERNAL_ASSERT(data.find(layerId) == data.end());
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

static void foreachTensorInplace(std::vector<IValue>& args, int64_t begin, int64_t end,
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
  // } else if (layer.key() == DispatchKey::Batched) {
  //   exclude = exclude.remove(DispatchKey::Batched);
  } else if (layer.key() == kBatchedKey) {
    exclude = exclude.remove(kBatchedKey);
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
    pushDynamicLayer(layer_.key());
  }

  bool prev_grad_enabled_;
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
  op.callBoxed(stack);

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
