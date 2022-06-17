// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/TensorWrapper.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <functorch/csrc/BatchRulesHelper.h>

#include <torch/library.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <torch/csrc/autograd/variable.h>
#include <c10/util/irange.h>
#include <ATen/FuncTorchTLS.h>
#include <iostream>

namespace at {
namespace functorch {

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
{
  if (transform_type == TransformType::Grad) {
    TORCH_INTERNAL_ASSERT(prev_grad_mode.has_value());
  }
  if (transform_type == TransformType::Jvp) {
    TORCH_INTERNAL_ASSERT(prev_fwd_grad_mode.has_value());
  }
  switch (transform_type) {
    case TransformType::Vmap:
      interpreter_ = Interpreter::Vmap(layerId, batchSize.value(), randomness.value());
      break;
    case TransformType::Grad:
      interpreter_ = Interpreter::Grad(layerId, prev_grad_mode.value());
      break;
    case TransformType::Jvp:
      interpreter_ = Interpreter::Jvp(layerId, prev_fwd_grad_mode.value());
      break;
    case TransformType::Functionalize:
      interpreter_ = Interpreter::Functionalize(layerId, functionalize_add_back_views.value());
      break;
    default:
      TORCH_INTERNAL_ASSERT(false);
  }
}

TransformType DynamicLayer::key() const {
  return interpreter_.key();
}

int64_t DynamicLayer::layerId() const {
  return interpreter_.level();
}

int64_t DynamicLayer::batchSize() const {
  return VmapInterpreterPtr(&interpreter_).batchSize();
}

RandomnessType DynamicLayer::randomness() const {
  return VmapInterpreterPtr(&interpreter_).randomness();
}

constexpr DispatchKeySet kFrontBackKeys({kDynamicLayerBackModeKey, kDynamicLayerFrontModeKey});

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
    TORCH_CHECK(dynamicLayerStack.size() == 0 || allow_inplace_requires_grad_,
        "You are attempting to call Tensor.requires_grad_() (or perhaps using ",
        "torch.autograd.functional.* APIs) inside of a function being transformed ",
        "by a functorch transform. ",
        "This is unsupported, please attempt to use the functorch transforms ",
        "(e.g. grad, vjp, jacrev, jacfwd, hessian) or call requires_grad_() "
        "outside of a function being transformed instead.");
  }
  void checkSupportsRetainGrad() const override {
    TORCH_CHECK(dynamicLayerStack.size() == 0,
        "You are attempting to call Tensor.retain_grad() ",
        "inside of a function being transformed ",
        "by a functorch transform. ",
        "This is unsupported, please attempt to use the functorch transforms ",
        "(e.g. grad, vjp, jacrev, jacfwd, hessian) or call retain_grad() "
        "outside of a function being transformed instead.");
  }

  std::vector<DynamicLayer> dynamicLayerStack;
  bool allow_inplace_requires_grad_ = false;
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

void setInplaceRequiresGradAllowed(bool allowed) {
  auto* functorch_tls = getRawFunctorchTLS();
  functorch_tls->allow_inplace_requires_grad_ = allowed;
}

bool getInplaceRequiresGradAllowed() {
  auto* functorch_tls = getRawFunctorchTLS();
  return functorch_tls->allow_inplace_requires_grad_;
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
    layer.interpreter().saveLocalDispatchKeySet(tmp);
  }
  ~SaveLocalDispatchKeySet() {
    auto& dynamicLayerStack = dynamicLayerStackAccessor();
    TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
    auto& layer = dynamicLayerStack.back();
    auto tmp = layer.interpreter().getSavedLocalDispatchKeySet();
    layer.interpreter().clearSavedLocalDispatchKeySet();
    c10::impl::_force_tls_local_dispatch_key_set(tmp);
  }
  SaveLocalDispatchKeySet(const SaveLocalDispatchKeySet&) = delete;
  SaveLocalDispatchKeySet& operator=(const SaveLocalDispatchKeySet&) = delete;
};

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

Tensor unwrapIfDead(const Tensor& tensor) {
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


#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
static void dump_local_tls() {
  auto tls = c10::impl::tls_local_dispatch_key_set();
  std::cout << "[Local Include] " << tls.included_ << std::endl;
  std::cout << "[Local Exclude] " << tls.excluded_ << std::endl;
}
#endif

struct WithoutTop {
  WithoutTop();
  ~WithoutTop();
  DynamicLayer layer_;
};

WithoutTop::WithoutTop(): layer_(popDynamicLayer()) {}
WithoutTop::~WithoutTop() {
  pushDynamicLayer(std::move(layer_));
}

// NOTE: [forward-mode AD decompositions hack]
//
// The mechanism is: in DynamicLayerFrontMode, IF we are dispatching on the
// jvp transform, AND we have a decomposition for the operation, then run
// the decomposition.
//
// Let's break that down. There are a douple of moving pieces.
//
// 0. How do we know what transform we're dispatching on?
// Easy, check the top of the DynamicLayerStack and read the transform.
//
// 1. Next, we must identify when an operation (e.g. nll_loss_backward)
// gets dispatched to.
// - register a special kernel to the DynamicLayerFrontMode key
//   (see JVP_DECOMP)
// - that special kernel invokes dynamicLayerFrontFallbackOperator with
//   an arg indicating we're going to use a decomp
//
// 2. Next, we need to call the decomposition. See call_decomposition_for_jvp.
// We currently use python decompositions that we torchscript.

// Ideally c10::OperatorHandle would have a field like this
// to identify the operator.
// The stuff here should map 1:1 with the operator name.
// aten::nll_loss_backward -> nll_loss_backward
// aten::add.Tensor -> add_Tensor

static void call_decomposition_for_jvp(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  run_jit_decomposition(op, stack);
}

static void dynamicLayerFrontFallbackOperator(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool decomp_jvp) {
  auto& dynamicLayerStack = dynamicLayerStackAccessor();
  TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
#ifdef HAS_TORCH_SHOW_DISPATCH_TRACE
  if (c10::show_dispatch_trace_enabled()) {
    std::cout << dynamicLayerStack << std::endl;
    dump_local_tls();
  }
#endif

  // Hack: if jvp and we have a decomposition registered, then do the decomposition
  if (dynamicLayerStack.back().interpreter().key() == TransformType::Jvp &&
      decomp_jvp) {
    return call_decomposition_for_jvp(op, stack);
  }

  // Save the current LocalDispatchKeySet (to the current DynamicLayer).
  // Upon exiting the current scope, that LocalDispatchKeySet gets restored.
  // When the current DynamicLayer dispatches to the next (inner) DynamicLayer,
  // it will also temporarily restore the saved LocalDispatchKeySet.
  SaveLocalDispatchKeySet guard;

  // Unwrap escaped GradWrappers
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(), unwrapIfDead);

  auto& layer = dynamicLayerStack.back();
  layer.interpreter().process(op, stack);
}

static c10::impl::ForceDispatchKeyGuard
restoreLocalDispatchKeySetRAII(const c10::impl::LocalDispatchKeySet& key_set) {
  return c10::impl::ForceDispatchKeyGuard(key_set);
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  return dynamicLayerFrontFallbackOperator(op, stack, false);
}

void dynamicLayerFrontFallBackWithDecomp(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  return dynamicLayerFrontFallbackOperator(op, stack, true);
}

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto& layer = dynamicLayerStackAccessor().back();
  auto restore_guard = restoreLocalDispatchKeySetRAII(layer.interpreter().getSavedLocalDispatchKeySet());
  WithoutTop guard;

  layer.interpreter().sendToNextInterpreter(op, stack);
}

TORCH_LIBRARY_IMPL(_, FT_DYNAMIC_LAYER_FRONT_MODE_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallback>());
}

TORCH_LIBRARY_IMPL(_, FT_DYNAMIC_LAYER_BACK_MODE_KEY, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerBackFallback>());
}

#define JVP_DECOMP(op) \
  m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallBackWithDecomp>());

#define JVP_DECOMP2(op, overload) \
  m.impl(#op "." #overload, torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallBackWithDecomp>());

TORCH_LIBRARY_IMPL(aten, FT_DYNAMIC_LAYER_FRONT_MODE_KEY, m) {
  JVP_DECOMP(nll_loss_backward);
  JVP_DECOMP(nll_loss2d_backward);
  JVP_DECOMP(_log_softmax_backward_data);
  JVP_DECOMP(_softmax_backward_data);
  OP_DECOMPOSE(log_sigmoid);
  JVP_DECOMP(log_sigmoid_forward);
  JVP_DECOMP(native_layer_norm_backward);
  JVP_DECOMP(native_batch_norm_backward);
  JVP_DECOMP(cudnn_batch_norm_backward);
}


}
} // namespace at
