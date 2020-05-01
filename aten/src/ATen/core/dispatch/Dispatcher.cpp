#include <ATen/core/dispatch/Dispatcher.h>
#include <list>
#include <sstream>

namespace c10 {

namespace detail {

class RegistrationListenerList final {
public:
  std::function<void()> addListener(std::unique_ptr<OpRegistrationListener> listener) {
    listeners_.push_back(std::move(listener));
    auto delete_it = --listeners_.end();
    return [this, delete_it] {
        listeners_.erase(delete_it);
    };
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
  std::list<std::unique_ptr<OpRegistrationListener>> listeners_;
};

void convert_to_ivalue_vector(std::vector<c10::IValue>& stack) {}
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

c10::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> c10::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return c10::nullopt;
    }
    return found->second;
  });
}

c10::optional<OperatorHandle> Dispatcher::findSchema(const OperatorName& overload_name) {
  auto it = findOp(overload_name);
  if (it.has_value()) {
    if (it->hasSchema()) {
      return it;
    } else {
      return c10::nullopt;
    }
  } else {
    return it;
  }
}

OperatorHandle Dispatcher::findSchemaOrThrow(const char* name, const char* overload_name) {
  auto it = findSchema({name, overload_name});
  if (!it.has_value()) {
    // Check if we have ANYTHING; if that's the case, that means you're
    // missing schema
    auto it2 = findOp({name, overload_name});
    if (!it2.has_value()) {
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name);
    } else {
      TORCH_CHECK(false, "Could not find schema for ", name, ".", overload_name,
        " but we found an implementation; did you forget to def() the operator?");
    }
  }
  return it.value();
}

// Postcondition: caller is responsible for disposing of registration when they
// are done
OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found != c10::nullopt) {
    return *found;
  }

  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
    operatorLookupTable.emplace(op_name, handle);
  });

  return handle;
}

RegistrationHandleRAII Dispatcher::registerLibrary(std::string ns, std::string debug) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = libraries_.find(ns);
  TORCH_CHECK(
    found == libraries_.end(),
    "Only a single TORCH_LIBRARY can be used to register the namespace ", ns,
    "; please put all of your definitions in a single TORCH_LIBRARY block.  "
    "If you were trying to specify implementations, consider using TORCH_LIBRARY_IMPL "
    "(which can be duplicated).  Previous registration of TORCH_LIBRARY was ",
    found->second, "; latest registration was ", debug
  );
  libraries_.emplace(ns, std::move(debug));
  return RegistrationHandleRAII([this, ns] {
    deregisterLibrary_(ns);
  });
}

void Dispatcher::deregisterLibrary_(const std::string& ns) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);
  libraries_.erase(ns);
}

RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  if (op.operatorIterator_->def_count == 0) {
    // NB: registerSchema is not idempotent! Only do it once!
    op.operatorIterator_->op.registerSchema(std::move(schema), std::move(debug));
    listeners_->callOnOperatorRegistered(op);
  } else {
    checkSchemaCompatibility(op, schema, debug);
  }

  // NB: do not increment the counts until AFTER error checking
  ++op.operatorIterator_->def_count;
  ++op.operatorIterator_->def_and_impl_count;

  return RegistrationHandleRAII([this, op, op_name] {
    deregisterDef_(op, op_name);
  });
}

void Dispatcher::checkSchemaCompatibility(const OperatorHandle& op, const FunctionSchema& schema, const std::string& debug) {
  TORCH_CHECK(op.schema() == schema, "Tried to register multiple operators with the same name and the same overload name but different schemas: ", schema, " (", debug, ") vs ", op.schema(), " (", op.debug(), ")");
  if (schema.isDefaultAliasAnalysisKind()) {
    // [BACKWARDS COMPAT] If the *new* schema is the default alias analysis
    // kind, for BC, we will accept it.  If we don't accept it, most extensions
    // that override existing operators will stop working (as they generally did
    // not specify alias information).
  } else if (op.schema().isDefaultAliasAnalysisKind()) {
    // [BACKWARDS COMPAT] If you POST-FACTO specify a non-default alias analysis
    // kind after we already have a schema for a function, bong it in for BC
    // reasons.
    op.operatorIterator_->op.updateSchemaAliasAnalysis(schema.aliasAnalysis());
  } else {
    TORCH_CHECK(op.schema().aliasAnalysis() == schema.aliasAnalysis(),
      "Tried to define the schema for ", toString(op.operator_name()), " with different alias analysis kinds: ",
      toString(op.schema().aliasAnalysis()), " (", op.debug(), ") vs ", toString(schema.aliasAnalysis()), " (", debug, ")");
  }
}

void Dispatcher::deregisterDef_(const OperatorHandle& op, const OperatorName& op_name) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  TORCH_INTERNAL_ASSERT(op.schema().operator_name() == op_name);

  // reduce def_count and actually deregister if no references left
  TORCH_INTERNAL_ASSERT(op.operatorIterator_->def_count > 0);
  TORCH_INTERNAL_ASSERT(op.operatorIterator_->def_and_impl_count > 0);

  --op.operatorIterator_->def_count;
  --op.operatorIterator_->def_and_impl_count;
  if (0 == op.operatorIterator_->def_count) {
    // note: call listeners *before* operator is removed, i.e. dispatcher is still valid for removed op
    // TODO: check that listeners are not relying on prepareForDeregistration()
    // invariant
    listeners_->callOnOperatorDeregistered(op);
    op.operatorIterator_->op.deregisterSchema();
  }

  cleanup(op, op_name);
}

RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorIterator_->op.registerKernel(dispatch_key, std::move(kernel), std::move(inferred_function_schema), std::move(debug));

  ++op.operatorIterator_->def_and_impl_count;

  return RegistrationHandleRAII([this, op, op_name, dispatch_key, handle] {
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}

void Dispatcher::deregisterImpl_(const OperatorHandle& op, const OperatorName& op_name, c10::optional<DispatchKey> dispatch_key, std::list<impl::OperatorEntry::KernelEntry>::iterator handle) {
  std::lock_guard<std::mutex> lock(mutex_);

  op.operatorIterator_->op.deregisterKernel_(dispatch_key, handle);

  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);

  TORCH_INTERNAL_ASSERT(op.operatorIterator_->def_and_impl_count > 0);
  --op.operatorIterator_->def_and_impl_count;

  cleanup(op, op_name);
}

// Test if the operator entry is completely dead, and if so remove it completely
void Dispatcher::cleanup(const OperatorHandle& op, const OperatorName& op_name) {
  if (0 == op.operatorIterator_->def_and_impl_count) {
    // TODO: rename this to "assert deregistration invariants"
    op.operatorIterator_->op.prepareForDeregistration();
    operators_.erase(op.operatorIterator_);
    operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
      operatorLookupTable.erase(op_name);
    });
  }
}

RegistrationHandleRAII Dispatcher::registerFallback(DispatchKey dispatchKey, KernelFunction kernel, std::string debug) {
  std::lock_guard<std::mutex> lock(mutex_);

  // TODO: fallbacks clobber each other completely unsafely, unlike regular
  // kernels
  backendFallbackKernels_.setKernel(dispatchKey, std::move(kernel));
  if (kernel.isFallthrough()) {
    backendsWithoutFallthrough_ = backendsWithoutFallthrough_.remove(dispatchKey);
  }

  return RegistrationHandleRAII([this, dispatchKey] {
    deregisterFallback_(dispatchKey);
  });
}

void Dispatcher::deregisterFallback_(DispatchKey dispatchKey) {
  std::lock_guard<std::mutex> lock(mutex_);

  backendFallbackKernels_.removeKernelIfExists(dispatchKey);
  backendsWithoutFallthrough_ = backendsWithoutFallthrough_.add(dispatchKey);
}


RegistrationHandleRAII Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    if (iter->def_count > 0) {
      listener->onOperatorRegistered(OperatorHandle(iter));
    }
  }

  auto removeListener = listeners_->addListener(std::move(listener));
  return RegistrationHandleRAII([this, removeListener] {
      std::lock_guard<std::mutex> lock(mutex_);
      removeListener();
  });
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

void Dispatcher::checkInvariants() const {
  for (const auto& op : operators_) {
    op.op.checkInvariants();
  }
  // NB: skip Undefined
  for (uint8_t i = 1; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys); i++) {
    auto k = static_cast<DispatchKey>(i);
    if (!backendsWithoutFallthrough_.has(k)) {
      const auto& kernel = backendFallbackKernels_[k];
      TORCH_INTERNAL_ASSERT(kernel.isFallthrough());
    }
  }
}

void Dispatcher::setManuallyBoxedKernelFor_(const OperatorHandle& op, KernelFunction::InternalBoxedKernelFunction* func) {
  op.operatorIterator_->op.setManuallyBoxedKernel_(func);
}

}
