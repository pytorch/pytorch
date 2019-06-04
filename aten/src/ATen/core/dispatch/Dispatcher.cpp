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
    return opDef.op.schema().name() == operator_name && opDef.op.schema().overload_name() == overload_name;
  });

  if (found == operators_.end()) {
    return c10::nullopt;
  }

  return OperatorHandle(found);
}

OperatorHandle Dispatcher::findOrRegisterSchema_(FunctionSchema&& schema, OperatorOptions&& options) {
  const auto found = findSchema(schema.name().c_str(), schema.overload_name().c_str());
  if (found != c10::nullopt) {
    if (found->schema() != schema) {
      std::ostringstream str;
      str << schema << " vs " << found->schema();
      TORCH_CHECK(false, "Tried to register multiple operators with the same name and the same overload name but different schemas: ", str.str());
    }
    if (found->options() != options) {
      TORCH_CHECK(false, "Tried to register multiple operators with the same schema but different options: ", toString(schema));
    }
    return *found;
  }

  operators_.emplace_back(std::move(schema), std::move(options));
  return OperatorHandle(--operators_.end());
}

SchemaRegistrationHandleRAII Dispatcher::registerSchema(FunctionSchema schema, OperatorOptions options) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  auto op = findOrRegisterSchema_(std::move(schema), std::move(options));

  ++op.operatorIterator_->refcount;
  if (1 == op.operatorIterator_->refcount) {
    // note: call listeners *after* operator is added, i.e. dispatcher is already valid for new op
    listeners_->callOnOperatorRegistered(op);
  }

  return SchemaRegistrationHandleRAII {op, RegistrationHandleRAII([this, op] {
    deregisterSchema_(op);
  })};
}

void Dispatcher::deregisterSchema_(const OperatorHandle& op) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  // reduce refcount and actually deregister if no references left
  TORCH_INTERNAL_ASSERT(op.operatorIterator_->refcount > 0);
  --op.operatorIterator_->refcount;
  if (0 == op.operatorIterator_->refcount) {
    op.operatorIterator_->op.prepareForDeregistration();

    // note: call listeners *before* operator is removed, i.e. dispatcher is still valid for removed op
    listeners_->callOnOperatorDeregistered(op);

    operators_.erase(op.operatorIterator_);
  }
}

RegistrationHandleRAII Dispatcher::registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func) {
  // note: this doesn't need the mutex to protect the iterator because write operations on the list keep iterators intact.
  return op.operatorIterator_->op.registerKernel(std::move(dispatch_key), DispatchTableEntry{kernel_func, std::move(cache_creator_func)});
}

RegistrationHandleRAII Dispatcher::registerCatchallKernel(const OperatorHandle& op, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func) {
  // note: this doesn't need the mutex to protect the iterator because write operations on the list keep iterators intact.
  return op.operatorIterator_->op.registerCatchallKernel(DispatchTableEntry{kernel_func, std::move(cache_creator_func)});
}

void Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    listener->onOperatorRegistered(OperatorHandle(iter));
  }

  listeners_->addListener(std::move(listener));
}

}
