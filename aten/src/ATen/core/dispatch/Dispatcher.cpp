#include <ATen/core/dispatch/Dispatcher.h>
#include <sstream>

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

c10::optional<OperatorHandle> Dispatcher::findSchema(const char* operator_name, const char* overload_name) {
  const auto found = std::find_if(operators_.begin(), operators_.end(), [&] (const OperatorDef& opDef) {
    return opDef.schema.name() == operator_name && opDef.schema.overload_name() == overload_name;
  });

  if (found == operators_.end()) {
    return c10::nullopt;
  }

  return OperatorHandle(found);
}

OperatorHandle Dispatcher::findOrRegisterSchema_(FunctionSchema&& schema) {
  const auto found = findSchema(schema.name().c_str(), schema.overload_name().c_str());
  if (found != c10::nullopt) {
    if (found->schema() != schema) {
      std::ostringstream str;
      str << schema << " vs " << found->schema();
      AT_ERROR("Tried to register multiple operators with the same name and the same overload name but different schemas: ", str.str());
    }
    return *found;
  }

  operators_.emplace_back(std::move(schema));
  return OperatorHandle(--operators_.end());
}

OperatorHandle Dispatcher::registerSchema(FunctionSchema schema) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  auto op = findOrRegisterSchema_(std::move(schema));

  ++op.operatorDefIterator_->refcount;
  if (1 == op.operatorDefIterator_->refcount) {
    // note: call listeners *after* operator is added, i.e. dispatcher is already valid for new op
    listeners_->callOnOperatorRegistered(op);
  }

  return op;
}

void Dispatcher::deregisterSchema(const OperatorHandle& op) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  // reduce refcount and actually deregister if no references left
  AT_ASSERT(op.operatorDefIterator_->refcount > 0);
  --op.operatorDefIterator_->refcount;
  if (0 == op.operatorDefIterator_->refcount) {
    if (!op.operatorDefIterator_->dispatchTable.isEmpty()) {
      std::ostringstream str;
      str << op.schema();
      AT_ERROR("Tried to deregister op schema for an operator that still has kernels registered. The operator schema is ", str.str());
    }

    // note: call listeners *before* operator is removed, i.e. dispatcher is still valid for removed op
    listeners_->callOnOperatorDeregistered(op);

    operators_.erase(op.operatorDefIterator_);
  }
}

void Dispatcher::registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.registerKernel(std::move(dispatch_key), DispatchTableEntry{kernel_func, std::move(cache_creator_func)});
}

void Dispatcher::deregisterKernel(const OperatorHandle& op, TensorTypeId dispatch_key) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.deregisterKernel(std::move(dispatch_key));
}

void Dispatcher::registerFallbackKernel(const OperatorHandle& op, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.registerFallbackKernel(DispatchTableEntry{kernel_func, std::move(cache_creator_func)});
}

void Dispatcher::deregisterFallbackKernel(const OperatorHandle& op) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.deregisterFallbackKernel();
}

void Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    listener->onOperatorRegistered(OperatorHandle(iter));
  }

  listeners_->addListener(std::move(listener));
}

}
