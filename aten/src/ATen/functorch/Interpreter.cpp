#include <ATen/functorch/Interpreter.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/VmapInterpreter.h>
#include <ATen/functorch/FunctionalizeInterpreter.h>
#include <ATen/functorch/ADInterpreters.h>
#include <ATen/functorch/DynamicLayer.h>

#include <utility>

namespace at { namespace functorch {

static DispatchKeySet get_all_dynlayer_keyset() {
  // NB: FULL_AFTER does not include the dispatch key

  // "all dispatch keys between DynamicLayer{Front, Back}Mode, inclusive"
  auto result =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::FuncTorchDynamicLayerFrontMode) -
    DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::FuncTorchDynamicLayerBackMode);
  result = result | DispatchKeySet({DispatchKey::FuncTorchDynamicLayerFrontMode});

  // Hack: don't handle the autocast dispatch keys. Their interaction with functorch
  // is weird.
  result = result - autocast_dispatch_keyset;

  // Hack: don't handle DispatchKey::FuncTorchVmapMode. We need a better way of modeling this.
  // In e.g. grad(vmap(f)), DispatchKey::FuncTorchVmapMode makes it so that all random operations,
  // even after we are done handling the vmap layer, error out.
  result = result.remove(DispatchKey::FuncTorchVmapMode);

  return result;
}

// TODO: This should be constexpr, but there are some methods
// of DispatchKeySet that haven't been marked constexpr yet.
static DispatchKeySet all_dynlayer_keyset = get_all_dynlayer_keyset();

static DispatchKeySet keysForEnteringDynamicLayer(TransformType key) {
  if (key == TransformType::Vmap) {
    // NB: Does not include DispatchKey::FuncTorchVmapMode. We may modulate the key when
    // constructing the DynamicLayer, but we don't control it when entering/exiting
    // the DynamicLayer.
    return DispatchKeySet({DispatchKey::FuncTorchBatched});
  } else if (key == TransformType::Grad || key == TransformType::Jvp) {
    return autograd_dispatch_keyset.add(DispatchKey::ADInplaceOrView);
  } else if (key == TransformType::Functionalize) {
    return DispatchKeySet(DispatchKey::Functionalize);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported key: ", key);
  }
}

DispatchKeySet keysToExcludeWhenEnteringDynamicLayer(TransformType key) {
  DispatchKeySet exclude = all_dynlayer_keyset;
  exclude = exclude.remove(DispatchKey::FuncTorchDynamicLayerBackMode);
  exclude = exclude - keysForEnteringDynamicLayer(key);
  return exclude;
}

void setup_dispatch_key_tls(TransformType key, DispatchKeySet also_include) {
  auto local_keyset = c10::impl::tls_local_dispatch_key_set();
  auto to_exclude = local_keyset.excluded_;
  to_exclude = to_exclude | keysToExcludeWhenEnteringDynamicLayer(key);
  to_exclude = to_exclude - keysForEnteringDynamicLayer(key);
  local_keyset.excluded_ = to_exclude;
  local_keyset.included_ = local_keyset.included_ | also_include;
  c10::impl::_force_tls_local_dispatch_key_set(local_keyset);
}

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

void sanityCheckStack(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto num_args = op.schema().arguments().size();
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(),
      [](const Tensor& tensor) {
        auto result = unwrapIfDead(tensor);
        auto* wrapper = maybeGetTensorWrapper(result);
        TORCH_INTERNAL_ASSERT(wrapper == nullptr);
        auto* batched = maybeGetBatchedImpl(std::move(result));
        TORCH_INTERNAL_ASSERT(batched == nullptr);
        return tensor;
      });
}

#define INTERPRETER_DISPATCH(type, method) \
  switch (key()) { \
    case TransformType::Vmap: \
      TORCH_INTERNAL_ASSERT(c10::holds_alternative<VmapInterpreterMeta>(this->meta()));\
      return VmapInterpreterPtr(this). method; \
    case TransformType::Grad: \
      TORCH_INTERNAL_ASSERT(c10::holds_alternative<GradInterpreterMeta>(this->meta()));\
      return GradInterpreterPtr(this). method; \
    case TransformType::Jvp: \
      TORCH_INTERNAL_ASSERT(c10::holds_alternative<JvpInterpreterMeta>(this->meta()));\
      return JvpInterpreterPtr(this). method; \
    case TransformType::Functionalize: \
      TORCH_INTERNAL_ASSERT(c10::holds_alternative<FunctionalizeInterpreterMeta>(this->meta()));\
      return FunctionalizeInterpreterPtr(this). method; \
    default: \
      TORCH_INTERNAL_ASSERT(false, "Unrecognized transform"); \
  }

void Interpreter::process(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  INTERPRETER_DISPATCH(key_, SINGLE_ARG(processImpl(op, stack)));
}

void Interpreter::sendToNextInterpreter(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case) {
  INTERPRETER_DISPATCH(key_, SINGLE_ARG(sendToNextInterpreterImpl(op, stack, grad_special_case)));
}

}}
