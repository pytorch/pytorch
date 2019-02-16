#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {

namespace detail {
class RegistrationListenerList final {
public:
  void addListener(std::unique_ptr<OpRegistrationListener> listener) {
    listeners_.push_back(std::move(listener));
  }

  void callOnOperatorRegistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorRegistered(op);
    }
  }

  void callOnOperatorDeregistered(const OperatorHandle& op) {
    for (auto& listener : listeners_) {
      listener->onOperatorDeregistered(op);
    }
  }
private:
  std::vector<std::unique_ptr<OpRegistrationListener>> listeners_;
};
}

OpRegistrationListener::~OpRegistrationListener() {}

Dispatcher::Dispatcher()
: operators_()
, listeners_(guts::make_unique<detail::RegistrationListenerList>())
, mutex_() {}

Dispatcher::~Dispatcher() {}

C10_EXPORT Dispatcher& Dispatcher::singleton() {
  static Dispatcher _singleton;
  return _singleton;
}

OperatorHandle Dispatcher::registerSchema(FunctionSchema schema) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  operators_.emplace_back(std::move(schema));
  auto op = OperatorHandle(--operators_.end());

  // note: call listeners *after* operator is added, i.e. dispatcher is already valid for new op
  listeners_->callOnOperatorRegistered(op);

  return op;
}

void Dispatcher::deregisterSchema(const OperatorHandle& op) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  if (!op.operatorDefIterator_->dispatchTable.isEmpty()) {
    AT_ERROR("Tried to deregister op schema that still has kernels registered");
  }

  // note: call listeners *before* operator is removed, i.e. dispatcher is still valid for removed op
  listeners_->callOnOperatorDeregistered(op);

  operators_.erase(op.operatorDefIterator_);
}

void Dispatcher::registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction* cache_creator_func) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.registerKernel(std::move(dispatch_key), DispatchTableEntry{kernel_func, cache_creator_func});
}

void Dispatcher::deregisterKernel(const OperatorHandle& op, TensorTypeId dispatch_key) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.deregisterKernel(dispatch_key);
}

void Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    listener->onOperatorRegistered(OperatorHandle(iter));
  }

  listeners_->addListener(std::move(listener));
}

}
