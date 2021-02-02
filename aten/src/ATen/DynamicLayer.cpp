#include <torch/library.h>
#include <ATen/DynamicLayer.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {

std::vector<DynamicLayer> dynamicLayerStack;

int64_t pushDynamicLayer(DispatchKey key) {
  auto layerId = 1 + dynamicLayerStack.size();
  dynamicLayerStack.emplace_back(key, layerId);

  if (layerId == 1) {
    std::cout << "DynamicLayer on" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);
  }

  return layerId;
}

DynamicLayer popDynamicLayer() {
  auto result = dynamicLayerStack.back();
  dynamicLayerStack.pop_back();

  if (dynamicLayerStack.size() == 0) {
    std::cout << "DynamicLayer off" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, false);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, false);
  }

  return result;
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if (dynamicLayerStack.size() == 0) {
    std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " start terminal" << std::endl;
    DispatchKeySet exclude;
    exclude = exclude.add(DispatchKey::DynamicLayerFront);
    exclude = exclude.add(DispatchKey::Batched);
    exclude = exclude.add(DispatchKey::Autograd);
    exclude = exclude.add(DispatchKey::AutogradOther);
    exclude = exclude.add(DispatchKey::AutogradCPU);
    exclude = exclude.add(DispatchKey::AutogradCUDA);
    exclude = exclude.add(DispatchKey::DynamicLayerBack);

    c10::impl::ExcludeDispatchKeyGuard guard(exclude);
    op.callBoxed(stack);
    std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " end terminal" << std::endl;
    return;
  }

  auto layer = dynamicLayerStack.back();

  DispatchKeySet exclude = DispatchKeySet::FULL;
  exclude = exclude.remove(DispatchKey::DynamicLayerBack);
  // NB: Alias dispatch key doesn't work in exclude set :(
  if (layer.key() == DispatchKey::Autograd) {
    std::cout << "enabling some autograd keys..." << std::endl;
    exclude = exclude.remove(DispatchKey::Autograd);
    exclude = exclude.remove(DispatchKey::AutogradOther);
    exclude = exclude.remove(DispatchKey::AutogradCPU);
    exclude = exclude.remove(DispatchKey::AutogradCUDA);
  } else {
    exclude = exclude.remove(layer.key());
  }
  c10::impl::ExcludeDispatchKeyGuard guard(exclude);
  // Exclude all keys except for layer.key and DynamicLayerBack
  // auto keyset = c10::impl::PODLocalDispatchKeySet();
  // keyset.set_excluded(exclude);
  // c10::impl::_force_tls_local_dispatch_key_set(keyset);

  std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " " << layer.key() << " " << dynamicLayerStack.size() << std::endl;

  // Re-dispatch
  op.callBoxed(stack);

  // Clear TLS
  // keyset = c10::impl::PODLocalDispatchKeySet();
  // c10::impl::_force_tls_local_dispatch_key_set(keyset);
  // c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
  // c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);
}

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "dynamicLayerBackFallback" << std::endl;

  // pop the top layer..
  auto handledLayer = popDynamicLayer();

  // "reset exclude set"
  auto keyset = c10::impl::PODLocalDispatchKeySet();
  c10::impl::_force_tls_local_dispatch_key_set(keyset);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);

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
