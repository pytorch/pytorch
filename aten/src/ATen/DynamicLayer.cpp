#include <torch/library.h>
#include <ATen/DynamicLayer.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {

std::vector<DynamicLayer> dynamicLayerStack;

int64_t pushDynamicLayer(DispatchKey key) {
  auto layerId = 1 + dynamicLayerStack.size();
  dynamicLayerStack.emplace_back(key, layerId);

  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);

  return layerId;
}

DynamicLayer popDynamicLayer() {
  auto result = dynamicLayerStack.back();
  dynamicLayerStack.pop_back();

  if (dynamicLayerStack.size() == 0) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, false);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, false);
  }

  return result;
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "dynamicLayerFrontFallback " << op.operator_name() << std::endl;
  if (dynamicLayerStack.size() == 0) {
    DispatchKeySet exclude;
    exclude = exclude.add(DispatchKey::DynamicLayerFront);
    exclude = exclude.add(DispatchKey::Batched);
    exclude = exclude.add(DispatchKey::Autograd);
    exclude = exclude.add(DispatchKey::DynamicLayerBack);

    c10::impl::ExcludeDispatchKeyGuard guard(exclude);
    op.callBoxed(stack);
    return;
  }

  auto layer = dynamicLayerStack.back();

  // Exclude all keys except for layer.key and DynamicLayerBack
  auto keyset = c10::impl::PODLocalDispatchKeySet();
  DispatchKeySet exclude = DispatchKeySet::FULL;
  exclude = exclude.remove(DispatchKey::DynamicLayerBack);
  exclude = exclude.remove(layer.key());
  keyset.set_excluded(exclude);
  c10::impl::_force_tls_local_dispatch_key_set(keyset);

  // Re-dispatch
  op.callBoxed(stack);

  // Clear TLS
  keyset = c10::impl::PODLocalDispatchKeySet();
  c10::impl::_force_tls_local_dispatch_key_set(keyset);
}

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "dynamicLayerBackFallback" << std::endl;

  // pop the top layer..
  auto handledLayer = popDynamicLayer();

  // "reset exclude set"
  auto keyset = c10::impl::PODLocalDispatchKeySet();
  c10::impl::_force_tls_local_dispatch_key_set(keyset);

  // Re-dispatch
  op.callBoxed(stack);

  // push the top layer back
  pushDynamicLayer(handledLayer.key());
}

TORCH_LIBRARY_IMPL(_, DynamicLayerFront, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallback>());
}

TORCH_LIBRARY_IMPL(_, DynamicLayerBack, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerBackFallback>());
}

} // namespace at
