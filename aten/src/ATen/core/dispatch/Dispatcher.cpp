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
, backendsWithoutFallthrough_(DispatchKeySet::FULL)
, listeners_(std::make_unique<detail::RegistrationListenerList>())
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

OperatorHandle Dispatcher::findSchemaOrThrow(const char* name, const char* overload_name) {
  return findSchema({name, overload_name}).value();
}

OperatorHandle Dispatcher::findOrRegisterSchema_(FunctionSchema&& schema, OperatorOptions&& options) {
  const auto found = findSchema(schema.operator_name());
  if (found != c10::nullopt) {
    if (found->schema() != schema) {
      TORCH_CHECK(false, "Tried to register multiple operators with the same name and the same overload name but different schemas: ", schema, " vs ", found->schema());
    }
    if (options.isDefaultAliasAnalysisKind()) {
      // just do nothing and let it pass.
    } else if (found->options().isDefaultAliasAnalysisKind()) {
      found->operatorIterator_->op.updateOptionsAliasAnalysis(options.aliasAnalysis());
    } else {
      TORCH_CHECK(
        found->options() == options,
        "Tried to register multiple operators with the same schema but different options: ", toString(schema));
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

std::pair<RegistrationHandleRAII, OperatorHandle> Dispatcher::registerSchema(FunctionSchema schema, OperatorOptions options) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  OperatorName op_name = schema.operator_name();

  auto op = findOrRegisterSchema_(std::move(schema), std::move(options));

  ++op.operatorIterator_->refcount;
  if (1 == op.operatorIterator_->refcount) {
    // note: call listeners *after* operator is added, i.e. dispatcher is already valid for new op
    listeners_->callOnOperatorRegistered(op);
  }

  return std::make_pair(RegistrationHandleRAII([this, op, op_name] {
    deregisterSchema_(op, op_name);
  }), op);
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

RegistrationHandleRAII Dispatcher::registerBackendFallbackKernel(DispatchKey dispatchKey, KernelFunction kernel) {
  auto inserted = backendFallbackKernels_.setKernel(dispatchKey, std::move(kernel));
  TORCH_CHECK(inserted == impl::KernelFunctionTable::SetKernelResult::ADDED_NEW_KERNEL, "Tried to register a backend fallback kernel for ", dispatchKey, " but there was already one registered.");
  if (kernel.isFallthrough()) {
    backendsWithoutFallthrough_ = backendsWithoutFallthrough_.remove(dispatchKey);
  }

  return RegistrationHandleRAII([this, dispatchKey] {
    deregisterBackendFallbackKernel_(dispatchKey);
  });
}

void Dispatcher::deregisterBackendFallbackKernel_(DispatchKey dispatchKey) {
  auto result = backendFallbackKernels_.removeKernelIfExists(dispatchKey);
  backendsWithoutFallthrough_ = backendsWithoutFallthrough_.add(dispatchKey);
  TORCH_INTERNAL_ASSERT(result == impl::KernelFunctionTable::RemoveKernelIfExistsResult::REMOVED_KERNEL, "Tried to deregister a backend fallback kernel for ", dispatchKey, " but there was none registered.");
}

RegistrationHandleRAII Dispatcher::registerKernel(const OperatorHandle& op, DispatchKey dispatch_key, KernelFunction kernel) {
  // note: this doesn't need the mutex to protect the iterator because write operations on the list keep iterators intact.
  return op.operatorIterator_->op.registerKernel(dispatch_key, std::move(kernel));
}

RegistrationHandleRAII Dispatcher::registerCatchallKernel(const OperatorHandle& op, KernelFunction kernel) {
  // note: this doesn't need the mutex to protect the iterator because write operations on the list keep iterators intact.
  return op.operatorIterator_->op.registerKernel(c10::nullopt, std::move(kernel));
}

void Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    listener->onOperatorRegistered(OperatorHandle(iter));
  }

  listeners_->addListener(std::move(listener));
}

[[noreturn]] void Dispatcher::reportError(const DispatchTable& dispatchTable, DispatchKey dispatchKey) {
  if (dispatchKey == DispatchKey::Undefined) {
    TORCH_CHECK(false,
          "There were no tensor arguments to this function (e.g., you passed an "
          "empty list of Tensors), but no fallback function is registered for schema ", dispatchTable.operatorName(),
          ".  This usually means that this function requires a non-empty list of Tensors.  "
          "Available functions are ", dispatchTable.listAllDispatchKeys())
  }

  const std::string dispatchKeyStr = toString(dispatchKey);
  TORCH_CHECK(false, "Could not run '", dispatchTable.operatorName(), "' with arguments",
          " from the '", dispatchKeyStr, "' backend. '",
          dispatchTable.operatorName(), "' is only available for these backends: ",
          dispatchTable.listAllDispatchKeys(), ".");
}

}
