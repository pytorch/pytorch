#include <torch/library.h>
#include <ATen/DynamicLayer.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/autograd/variable.h>

namespace at {

// Initial autograd layer, because autograd is always "on"
std::vector<DynamicLayer> dynamicLayerStack = { DynamicLayer(DispatchKey::Autograd, 1) };

DynmetaData kDynMetaDataSingleton;
std::mutex kDynMetaDataSingletonMutex;

DynmetaData& getGlobalDynmetaData() {
  return kDynMetaDataSingleton;
}

std::mutex& getGlobalDynmetaDataMutex() {
  return kDynMetaDataSingletonMutex;
}


bool gradLayerAtTop() {
  return dynamicLayerStack.back().key() == DispatchKey::Autograd;
}

optional<DynamicLayer> maybeCurrentDynamicLayer() {
  // NB: Exception for regular autograd, maybe tweak this
  if (dynamicLayerStack.size() <= 1) {
    return {};
  }
  return dynamicLayerStack.back();
}

std::vector<DynamicLayer>& getDynamicLayerStack() {
  return dynamicLayerStack;
}

int64_t pushDynamicLayer(DispatchKey key) {
  TORCH_INTERNAL_ASSERT(key != DispatchKey::Undefined);
  auto layerId = 1 + dynamicLayerStack.size();
  dynamicLayerStack.emplace_back(key, layerId);

  if (layerId == 2) {
    // std::cout << "DynamicLayer on" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);
  }

  return layerId;
}

DynamicLayer popDynamicLayer() {
  TORCH_INTERNAL_ASSERT(dynamicLayerStack.size() > 0);
  auto result = dynamicLayerStack.back();
  TORCH_INTERNAL_ASSERT(result.key() != DispatchKey::Undefined);
  dynamicLayerStack.pop_back();

  if (dynamicLayerStack.size() == 0) {
    // std::cout << "DynamicLayer off" << std::endl;
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, false);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, false);
  }

  return result;
}

DynamicLayer popDynamicLayerAndDeleteMetadata() {
  auto result = popDynamicLayer();
  auto level = result.layerId();

  // There is unfortunately a deadlock somewhere :/
  // std::lock_guard<std::mutex> guard(getGlobalDynmetaDataMutex());
  auto& data = getGlobalDynmetaData();
  auto it = data.find(level);
  if (it == data.end()) {
    return result;
  }
  for (auto& ptr : it->second) {
    auto val = ptr.lock();
    if (!val) continue;
    // Clear the unique_ptr inside the shared_ptr.
    (*val).reset();
  }
  // Clear the queue of weak_ptrs
  data[level].clear();

  return result;
}

void dynamicLayerFrontFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if (dynamicLayerStack.size() == 0) {
    // std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " terminal" << std::endl;
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
    // std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " end terminal" << std::endl;
    return;
  }

  auto layer = dynamicLayerStack.back();

  DispatchKeySet exclude = DispatchKeySet::FULL;
  exclude = exclude.remove(DispatchKey::DynamicLayerBack);
  // NB: Alias dispatch key doesn't work in exclude set :(
  if (layer.key() == DispatchKey::Autograd) {
    // std::cout << "enabling some autograd keys..." << std::endl;
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

  // std::cout << "dynamicLayerFrontFallback " << op.operator_name() << " " << layer.key() << " " << dynamicLayerStack.size() << std::endl;

  // Re-dispatch
  op.callBoxed(stack);

  // Clear TLS
  // keyset = c10::impl::PODLocalDispatchKeySet();
  // c10::impl::_force_tls_local_dispatch_key_set(keyset);
  // c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
  // c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);
}

struct WithoutTop {
  WithoutTop(): layer_(popDynamicLayer()) {
    // prev_grad_enabled_ = GradMode::is_enabled();
    // if (!prev_grad_enabled_) {
    //   GradMode::set_enabled(true);
    // }
  }
  ~WithoutTop() {
    pushDynamicLayer(layer_.key()); 
    // GradMode::set_enabled(prev_grad_enabled_);
  }

  bool prev_grad_enabled_;
  DynamicLayer layer_;
};

void dynamicLayerBackFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // std::cout << "dynamicLayerBackFallback" << std::endl;

  // pop the top layer. Put it back on dtor.
  WithoutTop guard;

  // "reset exclude set"
  // TODO: Still a problem with composabiilty and AutoNonVariableTypeGuard.
  auto keyset = c10::impl::PODLocalDispatchKeySet();
  c10::impl::_force_tls_local_dispatch_key_set(keyset);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerFront, true);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::DynamicLayerBack, true);

  // Re-dispatch
  op.callBoxed(stack);
}

TORCH_LIBRARY_IMPL(_, DynamicLayerFront, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerFrontFallback>());
}

TORCH_LIBRARY_IMPL(_, DynamicLayerBack, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dynamicLayerBackFallback>());
}

} // namespace at
