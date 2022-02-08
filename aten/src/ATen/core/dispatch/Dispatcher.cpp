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
}

OpRegistrationListener::~OpRegistrationListener()= default;

Dispatcher::Dispatcher()
: operators_()
, operatorLookupTable_()
, backendFallbackKernels_()
, listeners_(std::make_unique<detail::RegistrationListenerList>())
, mutex_() {}

Dispatcher::~Dispatcher() = default;

C10_EXPORT Dispatcher& Dispatcher::realSingleton() {
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

const std::vector<OperatorName> Dispatcher::getAllOpNames() {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::vector<OperatorName> {
    std::vector<OperatorName> allOpNames;
    for (const auto& op : operatorLookupTable) {
        allOpNames.push_back(op.first);
    }
    return allOpNames;
  });
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


// Adding explicit destructor definition in the cpp to over linker error in Windows builds.
// Windows build doesn't produce the destructor symbol in PyTorch libs
// causing a linker failure in downstream projects.
// x-ref https://github.com/pytorch/pytorch/issues/70032
OperatorHandle::~OperatorHandle() = default;

RegistrationHandleRAII Dispatcher::registerLibrary(std::string ns, std::string debug) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto found = libraries_.find(ns);
  TORCH_CHECK(
    found == libraries_.end(),
    "Only a single TORCH_LIBRARY can be used to register the namespace ", ns,
    "; please put all of your definitions in a single TORCH_LIBRARY block.  "
    "If you were trying to specify implementations, consider using TORCH_LIBRARY_IMPL "
    "(which can be duplicated).  If you really intended to define operators for a "
    "single namespace in a distributed way, you can use TORCH_LIBRARY_FRAGMENT to "
    "explicitly indicate this.  "
    "Previous registration of TORCH_LIBRARY was ",
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

  TORCH_CHECK(op.operatorDef_->def_count == 0, "Tried to register an operator (", schema, ") with the same name and overload name multiple times.",
                                                    " Each overload's schema should only be registered with a single call to def().",
                                                    " Duplicate registration: ", debug, ". Original registration: ", op.operatorDef_->op.debug());
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug));
  listeners_->callOnOperatorRegistered(op);

  // NB: do not increment the counts until AFTER error checking
  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;

  return RegistrationHandleRAII([this, op, op_name] {
    deregisterDef_(op, op_name);
  });
}

void Dispatcher::deregisterDef_(const OperatorHandle& op, const OperatorName& op_name) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  TORCH_INTERNAL_ASSERT(op.schema().operator_name() == op_name);

  // reduce def_count and actually deregister if no references left
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_count > 0);
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);

  --op.operatorDef_->def_count;
  --op.operatorDef_->def_and_impl_count;
  if (0 == op.operatorDef_->def_count) {
    // note: call listeners *before* operator is removed, i.e. dispatcher is still valid for removed op
    // TODO: check that listeners are not relying on prepareForDeregistration()
    // invariant
    listeners_->callOnOperatorDeregistered(op);
    op.operatorDef_->op.deregisterSchema();
  }

  cleanup(op, op_name);
}

RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    // NOLINTNEXTLINE(performance-move-const-arg)
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;

  return RegistrationHandleRAII([this, op, op_name, dispatch_key, handle] {
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}

void Dispatcher::deregisterImpl_(const OperatorHandle& op, const OperatorName& op_name, c10::optional<DispatchKey> dispatch_key, impl::OperatorEntry::AnnotatedKernelContainerIterator handle) {
  std::lock_guard<std::mutex> lock(mutex_);

  op.operatorDef_->op.deregisterKernel_(*this, dispatch_key, handle);

  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);

  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);
  --op.operatorDef_->def_and_impl_count;

  cleanup(op, op_name);
}

RegistrationHandleRAII Dispatcher::registerName(OperatorName op_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto op = findOrRegisterName_(op_name);
  ++op.operatorDef_->def_and_impl_count;
  return RegistrationHandleRAII(
      [this, op, op_name] { deregisterName_(op, op_name); });
}

void Dispatcher::deregisterName_(
    const OperatorHandle& op,
    const OperatorName& op_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);
  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);
  --op.operatorDef_->def_and_impl_count;
  cleanup(op, op_name);
}

// Test if the operator entry is completely dead, and if so remove it completely
void Dispatcher::cleanup(const OperatorHandle& op, const OperatorName& op_name) {
  if (0 == op.operatorDef_->def_and_impl_count) {
    // NOTE: Making this call fast is the only reason OperatorHandle
    // stores operatorIterator_!
    operators_.erase(op.operatorIterator_);
    operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
      operatorLookupTable.erase(op_name);
    });
  }
}

RegistrationHandleRAII Dispatcher::registerFallback(DispatchKey dispatchKey, KernelFunction kernel, std::string debug) {
  std::lock_guard<std::mutex> lock(mutex_);

  TORCH_CHECK(
    !backendFallbackKernels_[static_cast<uint8_t>(dispatchKey)].kernel.isValid(),
    "Tried to register multiple backend fallbacks for the same dispatch key ", dispatchKey, "; previous registration ",
    backendFallbackKernels_[static_cast<uint8_t>(dispatchKey)].debug, ", new registration ", debug
  );
  // NB: inferred function schema is always nullptr for fallbacks, as fallbacks
  // cannot be unobxed
  backendFallbackKernels_[static_cast<uint8_t>(dispatchKey)] = impl::AnnotatedKernel(std::move(kernel), nullptr, std::move(debug));

  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }

  return RegistrationHandleRAII([this, dispatchKey] {
    deregisterFallback_(dispatchKey);
  });
}

void Dispatcher::deregisterFallback_(DispatchKey dispatchKey) {
  std::lock_guard<std::mutex> lock(mutex_);

  backendFallbackKernels_[static_cast<uint8_t>(dispatchKey)] = {};

  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }
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

void Dispatcher::checkInvariants() const {
  for (const auto& op : operators_) {
    op.op.checkInvariants();
  }
}

std::vector<OperatorHandle> Dispatcher::findDanglingImpls() const {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::vector<OperatorHandle> {
    std::vector<OperatorHandle> opsWithDanglingImpls;
    for (const auto& op : operatorLookupTable) {
      if (!op.second.hasSchema()) {
        opsWithDanglingImpls.push_back(op.second);
      }
    }
    return opsWithDanglingImpls;
  });
}

std::vector<OperatorName> Dispatcher::getRegistrationsForDispatchKey(c10::optional<DispatchKey> k) const {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::vector<OperatorName> {
    std::vector<OperatorName> op_names;
    for (const auto& op : operatorLookupTable) {
      // If no DispatchKey is specified, print all of the operators.
      if (!k || op.second.hasKernelForDispatchKey(*k)) {
          op_names.push_back(op.first);
      }
    }
    return op_names;
  });
}

int64_t Dispatcher::sequenceNumberForRunningRecordFunction(DispatchKey dispatchKey) {
  int64_t seq_num = -1;
  // Setting sequence number in the Autograd case to associate
  // the forward range with the coresponding Autograd's node
  if (isIncludedInAlias(dispatchKey, DispatchKey::Autograd) && at::GradMode::is_enabled()) {
    seq_num = at::sequence_number::peek();
  }
  return seq_num;
}

void Dispatcher::runRecordFunction(at::RecordFunction& guard, const OperatorHandle& op, DispatchKey dispatchKey, const torch::jit::Stack &stack) {
  guard.before(op, c10::ArrayRef<const IValue>(stack.data(), stack.size()), sequenceNumberForRunningRecordFunction(dispatchKey));
}

void Dispatcher::runRecordFunction(at::RecordFunction& guard, const OperatorHandle& op, DispatchKey dispatchKey, torch::jit::Stack &&stack) {
  guard.before(op, c10::ArrayRef<const IValue>(stack.data(), stack.size()), sequenceNumberForRunningRecordFunction(dispatchKey));
}

void Dispatcher::runRecordFunction(at::RecordFunction& guard, const OperatorHandle& op, DispatchKey dispatchKey) {
  // Setting sequence number in the Autograd case to associate
  // the forward range with the coresponding Autograd's node
  guard.before(op, sequenceNumberForRunningRecordFunction(dispatchKey));
}

}
