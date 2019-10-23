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
, operatorLookupTable_()
, backendFallbackKernels_()
, listeners_(guts::make_unique<detail::RegistrationListenerList>())
, mutex_() {}

Dispatcher::~Dispatcher() {}

C10_EXPORT Dispatcher& Dispatcher::singleton() {
  static Dispatcher _singleton;
  return _singleton;
}

c10::optional<OperatorHandle> Dispatcher::findSchema(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> c10::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return c10::nullopt;
    }
    return found->second;
  });
}

OperatorHandle Dispatcher::findOrRegisterSchema_(FunctionSchema&& schema, OperatorOptions&& options) {
  const auto found = findSchema(schema.operator_name());
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

  OperatorName op_name = schema.operator_name();
  operators_.emplace_back(std::move(schema), std::move(options));
  OperatorHandle handle(--operators_.end());
  operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
    operatorLookupTable.emplace(op_name, handle);
  });

  return handle;
}

SchemaRegistrationHandleRAII Dispatcher::registerSchema(FunctionSchema schema, OperatorOptions options) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  OperatorName op_name = schema.operator_name();

  auto op = findOrRegisterSchema_(std::move(schema), std::move(options));

  ++op.operatorIterator_->refcount;
  if (1 == op.operatorIterator_->refcount) {
    // note: call listeners *after* operator is added, i.e. dispatcher is already valid for new op
    listeners_->callOnOperatorRegistered(op);
  }

  return SchemaRegistrationHandleRAII {op, RegistrationHandleRAII([this, op, op_name] {
    deregisterSchema_(op, op_name);
  })};
}

void Dispatcher::deregisterSchema_(const OperatorHandle& op, const OperatorName& op_name) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  TORCH_INTERNAL_ASSERT(op.schema().operator_name() == op_name);

  // reduce refcount and actually deregister if no references left
  TORCH_INTERNAL_ASSERT(op.operatorIterator_->refcount > 0);
  --op.operatorIterator_->refcount;
  if (0 == op.operatorIterator_->refcount) {
    op.operatorIterator_->op.prepareForDeregistration();

    // note: call listeners *before* operator is removed, i.e. dispatcher is still valid for removed op
    listeners_->callOnOperatorDeregistered(op);

    operators_.erase(op.operatorIterator_);
    operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
      operatorLookupTable.erase(op_name);
    });
  }
}

RegistrationHandleRAII Dispatcher::registerBackendFallbackKernel(TensorTypeId dispatchKey, KernelFunction kernel) {
  backendFallbackKernels_.write([&] (ska::flat_hash_map<TensorTypeId, KernelFunction>& backendFallbackKernels) {
    auto inserted = backendFallbackKernels.emplace(dispatchKey, std::move(kernel));
    TORCH_CHECK(inserted.second, "Tried to register a backend fallback kernel for ", dispatchKey, " but there was already one registered.");
  });

  return RegistrationHandleRAII([this, dispatchKey] {
    deregisterBackendFallbackKernel_(dispatchKey);
  });
}

void Dispatcher::deregisterBackendFallbackKernel_(TensorTypeId dispatchKey) {
  backendFallbackKernels_.write([&] (ska::flat_hash_map<TensorTypeId, KernelFunction>& backendFallbackKernels) {
    size_t numRemoved = backendFallbackKernels.erase(dispatchKey);
    TORCH_INTERNAL_ASSERT(1 == numRemoved, "Tried to deregister a backend fallback kernel for ", dispatchKey, " but there was none registered.");
  });
}

RegistrationHandleRAII Dispatcher::registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction kernel) {
  // note: this doesn't need the mutex to protect the iterator because write operations on the list keep iterators intact.
  return op.operatorIterator_->op.registerKernel(std::move(dispatch_key), std::move(kernel));
}

RegistrationHandleRAII Dispatcher::registerCatchallKernel(const OperatorHandle& op, KernelFunction kernel) {
  // note: this doesn't need the mutex to protect the iterator because write operations on the list keep iterators intact.
  return op.operatorIterator_->op.registerCatchallKernel(std::move(kernel));
}

void Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    listener->onOperatorRegistered(OperatorHandle(iter));
  }

  listeners_->addListener(std::move(listener));
}

}
