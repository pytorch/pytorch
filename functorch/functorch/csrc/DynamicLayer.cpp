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
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/csrc/autograd/variable.h>
#include <c10/util/irange.h>
#include <ATen/FuncTorchTLS.h>

namespace at {
namespace functorch {

std::ostream& operator<<(std::ostream& os, const TransformType& t) {
  switch (t) {
    case TransformType::Torch:
      os << "Torch";
      break;
    case TransformType::Vmap:
      os << "Vmap";
      break;
    case TransformType::Grad:
      os << "Grad";
      break;
    case TransformType::Jvp:
      os << "Jvp";
      break;
    case TransformType::Functionalize:
      os << "Functionalize";
      break;
    default:
      TORCH_INTERNAL_ASSERT(false);
  }
  return os;
}

constexpr DispatchKeySet all_dynlayer_keyset = DispatchKeySet({
  kDynamicLayerFrontModeKey,
  kDynamicLayerBackModeKey,
  kGradWrapperKey,
  DispatchKey::Functionalize,
  // DispatchKey::Batched,
  kBatchedKey,
  DispatchKey::PythonTLSSnapshot,
  DispatchKey::ADInplaceOrView
}) | autograd_dispatch_keyset;

void setDynamicLayerFrontBackKeysIncluded(bool included) {
  c10::impl::tls_set_dispatch_key_included(kDynamicLayerFrontModeKey, included);
  c10::impl::tls_set_dispatch_key_included(kDynamicLayerBackModeKey, included);
}

DynamicLayer::DynamicLayer(
    TransformType transform_type,
    int64_t layerId,
    optional<int64_t> batchSize,
    optional<RandomnessType> randomness,
    optional<bool> prev_grad_mode,
    optional<bool> prev_fwd_grad_mode,
    optional<bool> functionalize_add_back_views)
  :
    transform_type_(transform_type),
    layerId_(layerId),
    batchSize_(batchSize),
    randomness_(randomness),
    prevGradMode_(prev_grad_mode),
    prevFwdGradMode_(prev_fwd_grad_mode),
    functionalizeAddBackViews_(functionalize_add_back_views)
{
  if (transform_type == TransformType::Grad) {
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }
  if (transform_type == TransformType::Jvp) {
    TORCH_INTERNAL_ASSERT(prev_fwd_grad_mode.has_value());
  }
}

TransformType DynamicLayer::key() const {
  return transform_type_;
}

int64_t DynamicLayer::layerId() const {
  return layerId_;
}

int64_t DynamicLayer::batchSize() const {
  TORCH_INTERNAL_ASSERT(batchSize_);
  return *batchSize_;
}

RandomnessType DynamicLayer::randomness() const {
  TORCH_INTERNAL_ASSERT(randomness_);
  return *randomness_;
}

optional<bool> DynamicLayer::prevGradMode() const {
  return prevGradMode_;
}

optional<bool> DynamicLayer::prevFwdGradMode() const {
  return prevFwdGradMode_;
}

void DynamicLayer::saveLocalDispatchKeySet(c10::impl::LocalDispatchKeySet keyset) {
  TORCH_INTERNAL_ASSERT(!savedLocalDispatchKeySet_.has_value());
  savedLocalDispatchKeySet_ = std::move(keyset);
}

void DynamicLayer::clearSavedLocalDispatchKeySet() {
  TORCH_INTERNAL_ASSERT(savedLocalDispatchKeySet_.has_value());
  savedLocalDispatchKeySet_ = c10::nullopt;
}

c10::impl::LocalDispatchKeySet DynamicLayer::getSavedLocalDispatchKeySet() const {
  TORCH_INTERNAL_ASSERT(savedLocalDispatchKeySet_.has_value());
  return *savedLocalDispatchKeySet_;
}

constexpr DispatchKeySet kFrontBackKeys({kDynamicLayerBackModeKey, kDynamicLayerFrontModeKey});

optional<bool> DynamicLayer::functionalizeAddBackViews() const {
  return functionalizeAddBackViews_;
}

using DynmetaData = std::unordered_map<int64_t, std::shared_ptr<bool>>;
DynmetaData kDynMetaDataSingleton;

static DynmetaData& getGlobalDynmetaData() {
  return kDynMetaDataSingleton;
}

class FuncTorchTLS : public FuncTorchTLSBase {
 public:
  FuncTorchTLS() {}

  std::unique_ptr<FuncTorchTLSBase> deepcopy() const override {
    auto result = std::make_unique<FuncTorchTLS>();
    result->dynamicLayerStack = dynamicLayerStack;
    return result;
  }

  int64_t checkSupportsAutogradFunction() const override {
    TORCH_CHECK(dynamicLayerStack.size() == 0,
        "functorch functions (vmap, grad, vjp, etc.) currently do not support the use of autograd.Function. ",
        "Please rewrite your function to not use autograd.Function while we work on fixing this");
    return 0;
  }

  void checkSupportsInplaceRequiresGrad() const override {
    // Does nothing
  }
  void checkSupportsRetainGrad() const override {
    // Does nothing
  }

  std::vector<DynamicLayer> dynamicLayerStack;
};

static FuncTorchTLS* getRawFunctorchTLS() {
  auto& state = functorchTLSAccessor();
  if (state == nullptr) {
    state = std::make_unique<FuncTorchTLS>();
  }
  // Raw pointer usage OK, `state` keeps the pointer alive
  FuncTorchTLSBase* raw_state = state.get();
  FuncTorchTLS* result = static_cast<FuncTorchTLS*>(raw_state);
  return result;
}

static std::vector<DynamicLayer>& dynamicLayerStackAccessor() {
  return getRawFunctorchTLS()->dynamicLayerStack;
}

std::shared_ptr<bool> getLifeHandleForLevel(int64_t level) {
  auto it = getGlobalDynmetaData().find(level);
  TORCH_INTERNAL_ASSERT(it != kDynMetaDataSingleton.end(), "level should be alive");
  return it->second;
}

optional<DynamicLayer> maybeCurrentDynamicLayer() {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  if (dynamicLayerStack.size() == 0) {
    return {};
  }
  return dynamicLayerStack.back();
}

struct SaveLocalDispatchKeySet {
 public:
  SaveLocalDispatchKeySet() {
    auto& dynamicLayerStack = dynamicLayerStackAccessor();
    TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
    auto& layer = dynamicLayerStack.back();
    auto tmp = c10::impl::tls_local_dispatch_key_set();
    layer.saveLocalDispatchKeySet(tmp);
  }
  ~SaveLocalDispatchKeySet() {
    auto& dynamicLayerStack = dynamicLayerStackAccessor();
    TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
    auto& layer = dynamicLayerStack.back();
    auto tmp = layer.getSavedLocalDispatchKeySet();
    layer.clearSavedLocalDispatchKeySet();
    c10::impl::_force_tls_local_dispatch_key_set(tmp);
  }
  SaveLocalDispatchKeySet(const SaveLocalDispatchKeySet&) = delete;
  SaveLocalDispatchKeySet& operator=(const SaveLocalDispatchKeySet&) = delete;
};

static c10::impl::ForceDispatchKeyGuard
restoreLocalDispatchKeySetRAII(const DynamicLayer& layer) {
  auto tmp = layer.getSavedLocalDispatchKeySet();
  return c10::impl::ForceDispatchKeyGuard(tmp);
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
  dynamicLayerStack.pop_back();

  if (dynamicLayerStack.size() == 0) {
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
    if (c10::show_dispatch_trace_enabled()) {
      std::cout << "DynamicLayer off" << std::endl;
    }
#endif
    setDynamicLayerFrontBackKeysIncluded(false);
  }

  return result;
}

static int64_t pushDynamicLayer(DynamicLayer&& dynamic_layer) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  int64_t layerId = 1 + dynamicLayerStack.size();
  TORCH_INTERNAL_ASSERT(layerId == dynamic_layer.layerId());
  dynamicLayerStack.emplace_back(dynamic_layer);

  if (layerId == 1) {
    setDynamicLayerFrontBackKeysIncluded(true);
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
    if (c10::show_dispatch_trace_enabled()) {
      std::cout << "DynamicLayer on" << std::endl;
    }
#endif
  }

  return layerId;
}

int64_t initAndPushDynamicLayer(
    TransformType transform_type,
    optional<int64_t> batch_size,
    optional<RandomnessType> randomness,
    optional<bool> prev_grad_mode,
    optional<bool> prev_fwd_grad_mode,
    optional<bool> functionalize_add_back_views) {
  const auto& dynamicLayerStack = dynamicLayerStackAccessor();
  const auto layerId = 1 + dynamicLayerStack.size();
  DynamicLayer new_layer(transform_type, layerId, batch_size, randomness, prev_grad_mode, prev_fwd_grad_mode, functionalize_add_back_views);
  pushDynamicLayer(std::move(new_layer));

  auto& data = getGlobalDynmetaData();

  TORCH_INTERNAL_ASSERT(data.find(layerId) == data.end());
  if (transform_type == TransformType::Grad) {
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }
  if (transform_type == TransformType::Jvp) {
    TORCH_INTERNAL_ASSERT(prev_fwd_grad_mode.has_value());
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
  if (dynlayerStack.back().key() != TransformType::Grad && dynlayerStack.back().key() != TransformType::Jvp) {
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

std::ostream& operator<< (std::ostream& os, const DynamicLayer& layer) {
  os << layer.layerId() << ":" << layer.key();
  return os;
}
std::ostream& operator<< (std::ostream& os, const std::vector<DynamicLayer>& dls) {
  os << "DynamicLayerStack[ ";
  for (const auto& layer : dls) {
    os << layer << " ";
  }
  os << "]";
  return os;
}

void sanityCheckNotFunctional(const c10::OperatorHandle& op, torch::jit::Stack* stack, size_t num_args) {
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(),
      [](const Tensor& tensor) {
        TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(tensor));
        return tensor;
      });
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
  if (dynamicLayerStack.back().key() != TransformType::Grad && dynamicLayerStack.back().key() != TransformType::Jvp) {
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

static DispatchKeySet keysForEnteringDynamicLayer(TransformType key) {
  if (key == TransformType::Vmap) {
    // NB: Does not include kVmapModeKey. We may modulate the key when
    // constructing the DynamicLayer, but we don't control it when entering/exiting
    // the DynamicLayer.
    return DispatchKeySet({kBatchedKey});
  } else if (key == TransformType::Grad || key == TransformType::Jvp) {
    return autograd_dispatch_keyset.add(DispatchKey::ADInplaceOrView);
  } else if (key == TransformType::Functionalize) {
    return DispatchKeySet(DispatchKey::Functionalize);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported key: ", key);
  }
}

#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
static void dump_local_tls() {
  auto tls = c10::impl::tls_local_dispatch_key_set();
  std::cout << "[Local Include] " << tls.included_ << std::endl;
  std::cout << "[Local Exclude] " << tls.excluded_ << std::endl;
}
#endif

static DispatchKeySet keysToExcludeWhenEnteringDynamicLayer(TransformType key) {
  DispatchKeySet exclude = all_dynlayer_keyset;
  exclude = exclude.remove(kDynamicLayerBackModeKey);
  exclude = exclude - keysForEnteringDynamicLayer(key);
  return exclude;
}
static bool isFunctionalTensorAtCurrentLevel(const Tensor& tensor) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  auto layer = dynamicLayerStack.back();
  auto level = layer.layerId();

  if (!at::functionalization::impl::isFunctionalTensor(tensor)) {
    return false;
  }
  const auto* functional = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
  auto functional_level = functional->level();
  return functional_level == level;
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
  if (c10::show_dispatch_trace_enabled()) {
    std::cout << dynamicLayerStack << std::endl;
    dump_local_tls();
  }
#endif

  // Save the current LocalDispatchKeySet (to the current DynamicLayer).
  // Upon exiting the current scope, that LocalDispatchKeySet gets restored.
  // When the current DynamicLayer dispatches to the next (inner) DynamicLayer,
  // it will also temporarily restore the saved LocalDispatchKeySet.
  TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
  SaveLocalDispatchKeySet guard;

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

  auto& layer = dynamicLayerStack.back();

  DispatchKeySet exclude = keysToExcludeWhenEnteringDynamicLayer(layer.key());
  DispatchKeySet hacky_include;

  bool functionalization_add_back_views = false;

  // hack
  if (layer.key() == TransformType::Vmap) {
    hacky_include = hacky_include.add(kVmapModeKey);
  } else if (layer.key() == TransformType::Functionalize) {
    // We always want to call the functionalization kernels if functionalize() is on the layer stack.
    // It's the responsibility of the functionalization kernel to no-op and redispatch
    // if none of the input tensors are functional.
    hacky_include = hacky_include | DispatchKeySet({DispatchKey::Functionalize});
    functionalization_add_back_views = layer.functionalizeAddBackViews().has_value() && *(layer.functionalizeAddBackViews());
  }
  auto local_keyset = c10::impl::tls_local_dispatch_key_set();
  local_keyset.excluded_ = local_keyset.excluded_ | exclude;
  local_keyset.included_ = local_keyset.included_ | hacky_include;
  c10::impl::_force_tls_local_dispatch_key_set(local_keyset);
  // Only matters for functionalization.
  // We have some side-car TLS that we can set to toggle the functionaliation behavior.
  // If set, then we functionalization will only remove mutations, instead of
  // removing both mutations AND view operators.
  at::functionalization::impl::FunctionalizationReapplyViewsGuard functional_guard(functionalization_add_back_views);

#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
  if (c10::show_dispatch_trace_enabled()) {
    dump_local_tls();
  }
#endif

  // Re-dispatch
  op.callBoxed(stack);
  auto ret_size = op.schema().returns().size();
  foreachTensorInplace(*stack, stack->size() - ret_size, stack->size(),
    [&](const Tensor& tensor) {
      if (at::functionalization::impl::isFunctionalTensor(tensor)) {
        auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
        // Functorch is responsible for setting the level on the wrapper, since we don't
        // have that info available in core (for now).
        // We could just "propagate" the level from the input tensors inside of the functionalize kernels,
        // but unfortunately we can't do that for factory operators.
        wrapper->set_level(layer.layerId());
      }
      return tensor;
    }
  );
}

struct WithoutTop {
  WithoutTop(): layer_(popDynamicLayer()) {
  }
  ~WithoutTop() {
    pushDynamicLayer(std::move(layer_));
  }

  DynamicLayer layer_;
};

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto cur_level = getDynamicLayerStack().back().layerId();
  auto cur_key = getDynamicLayerStack().back().key();

  optional<bool> prev_grad_mode = getDynamicLayerStack().back().prevGradMode();
  optional<bool> prev_fwd_grad_mode = getDynamicLayerStack().back().prevFwdGradMode();
  if (cur_key == TransformType::Grad) {
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }
  if (cur_key == TransformType::Jvp) {
    TORCH_INTERNAL_ASSERT(prev_fwd_grad_mode.has_value());
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
  auto args_size = op.schema().arguments().size();
  if (cur_key == TransformType::Grad || cur_key == TransformType::Jvp) {
    auto args_size = op.schema().arguments().size();
    // Step 1
    auto front = stack->size() - args_size;
    for (const auto arg_idx : c10::irange(0, args_size)) {
      stack->push_back((*stack)[front + arg_idx]);
    }
    // Step 2
    foreachTensorInplace(*stack, stack->size() - args_size, stack->size(), unwrap);
  } else if (cur_key == TransformType::Functionalize) {
    // For now, we don't support nested functionalization calls.
    // This check just enforces that - after the functionalize kernel runs
    // and we hit the BackModeFallback, we'll have unwrapped our FunctionalTensors
    // so we can check that the unwrapped thing is not another (nested) FunctionalTensor.
    sanityCheckNotFunctional(op, stack, args_size);
  }

  auto restore_guard = restoreLocalDispatchKeySetRAII(getDynamicLayerStack().back());

  // pop the top layer. Put it back on dtor.
  WithoutTop guard;

#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
  if (c10::show_dispatch_trace_enabled()) {
    dump_local_tls();
  }
#endif

  // See NOTE [grad and vjp interaction with no_grad]
  optional<c10::AutoGradMode> grad_guard;
  if (cur_key == TransformType::Grad && prev_grad_mode.has_value() && *prev_grad_mode == false) {
    grad_guard.emplace(*prev_grad_mode);
  }
  optional<c10::AutoFwGradMode> fw_grad_guard;
  if (cur_key == TransformType::Jvp &&
      prev_fwd_grad_mode.has_value() && prev_fwd_grad_mode.value() == false) {
    fw_grad_guard.emplace(*prev_fwd_grad_mode);
  }

  // Re-dispatch
  if (dynamicLayerStackAccessor().size() == 0) {
    sanityCheckStack(op, stack);
  }
  op.callBoxed(stack);

  // Step 4, 5, 6
  auto ret_size = op.schema().returns().size();
  if (cur_key == TransformType::Grad || cur_key == TransformType::Jvp) {
    // Step 4
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
  } else if (cur_key == TransformType::Functionalize) {
    sanityCheckNotFunctional(op, stack, ret_size);
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
