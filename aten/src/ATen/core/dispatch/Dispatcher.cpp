#include <ATen/core/PythonOpRegistrationTrampoline.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <list>
#include <utility>

#include <c10/util/env.h>
#ifdef FBCODE_CAFFE2
#include <c10/util/static_tracepoint.h>
#endif

namespace c10 {

#ifdef FBCODE_CAFFE2
TORCH_SDT_DEFINE_SEMAPHORE(operator_start)
TORCH_SDT_DEFINE_SEMAPHORE(operator_end)
#endif

bool show_dispatch_trace() {
  static const auto envar = c10::utils::get_env("TORCH_SHOW_DISPATCH_TRACE");

  if (envar.has_value()) {
    if (envar == "0") {
      return false;
    }
    if (envar == "1") {
      return true;
    }
    TORCH_WARN(
        "ignoring invalid value for TORCH_SHOW_DISPATCH_TRACE: ",
        envar.value(),
        " valid values are 0 or 1.");
  }

  return false;
}

static thread_local int64_t dispatch_trace_nesting_value_;

void dispatch_trace_nesting_incr() { ++dispatch_trace_nesting_value_; }
void dispatch_trace_nesting_decr() { --dispatch_trace_nesting_value_; }
int64_t dispatch_trace_nesting_value() { return dispatch_trace_nesting_value_; }

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

void _print_dispatch_trace(const std::string& label, const std::string& op_name, const DispatchKeySet& dispatchKeySet) {
  auto nesting_value = dispatch_trace_nesting_value();
  for (int64_t i = 0; i < nesting_value; ++i) std::cerr << " ";
  std::cerr << label << " op=[" << op_name << "], key=[" << toString(dispatchKeySet.highestPriorityTypeId()) << "]" << std::endl;
}
} // namespace detail

OpRegistrationListener::~OpRegistrationListener()= default;

Dispatcher::Dispatcher()
: operators_()
, operatorLookupTable_()
, backendFallbackKernels_()
, listeners_(std::make_unique<detail::RegistrationListenerList>())
, cond_var_()
, guard_(std::make_shared<Guard>())
{}

Dispatcher::~Dispatcher() {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  guard_->alive.store(false);
}

C10_EXPORT Dispatcher& Dispatcher::realSingleton() {
  static Dispatcher _singleton;
  return _singleton;
}

std::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read([&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> std::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return std::nullopt;
    }
    return found->second;
  });
}

// NB: If you add more waitFor* implementations, you also have to add
// appropriate notify_all() calls to the relevant register calls

void Dispatcher::waitForDef(const FunctionSchema& schema) {
  using namespace std::chrono_literals;
  std::unique_lock<std::mutex> lock(guard_->mutex);
  bool r = cond_var_.wait_for(lock, 2s, [&]{
    return findOp(schema.operator_name()) != std::nullopt;
  });
  TORCH_INTERNAL_ASSERT(r,
    "Expected main interpreter to define ", schema.operator_name(),
    ", but this didn't happen within timeout.  Are you trying to load "
    "different models in the same torchdeploy/multipy instance?  You "
    "must warmup each interpreter identically, e.g., import all "
    "the same dependencies.");
}

void Dispatcher::waitForImpl(const OperatorName& op_name, std::optional<c10::DispatchKey> maybe_dk) {
  using namespace std::chrono_literals;
  std::unique_lock<std::mutex> lock(guard_->mutex);
  auto dk = maybe_dk.value_or(DispatchKey::CompositeImplicitAutograd);
  auto op = findOrRegisterName_(op_name);
  bool r = cond_var_.wait_for(lock, 2s, [&]{
    // NB: this is slightly unsound for overrides, but overrides are
    // funny business anyway
    return op.hasKernelForDispatchKey(dk);
  });
  TORCH_INTERNAL_ASSERT(r,
    "Expected main interpreter to implement ", dk, " for ", op_name,
    ", but this didn't happen within timeout.  Are you trying to load "
    "different models in the same torchdeploy/multipy instance?  You "
    "must warmup each interpreter identically, e.g., import all "
    "the same dependencies.");
}

std::optional<OperatorHandle> Dispatcher::findSchema(const OperatorName& overload_name) {
  auto it = findOp(overload_name);
  if (it.has_value()) {
    if (it->hasSchema()) {
      return it;
    } else {
      return std::nullopt;
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
  if (found != std::nullopt) {
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
  std::lock_guard<std::mutex> lock(guard_->mutex);
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
  return RegistrationHandleRAII([guard = this->guard_, this, ns] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterLibrary_(ns);
  });
}

void Dispatcher::deregisterLibrary_(const std::string& ns) {
  // we need a lock to avoid concurrent writes
  libraries_.erase(ns);
}

RegistrationHandleRAII Dispatcher::registerDef(FunctionSchema schema, std::string debug, std::vector<at::Tag> tags) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(guard_->mutex);

  OperatorName op_name = schema.operator_name();
  auto op = findOrRegisterName_(op_name);

  TORCH_CHECK(op.operatorDef_->def_count == 0, "Tried to register an operator (", schema, ") with the same name and overload name multiple times.",
                                                    " Each overload's schema should only be registered with a single call to def().",
                                                    " Duplicate registration: ", debug, ". Original registration: ", op.operatorDef_->op.debug());
  op.operatorDef_->op.registerSchema(std::move(schema), std::move(debug), std::move(tags));
  listeners_->callOnOperatorRegistered(op);

  // NB: do not increment the counts until AFTER error checking
  ++op.operatorDef_->def_count;
  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name] {
    // we need a lock to avoid concurrent writes
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterDef_(op, op_name);
  });
}

void Dispatcher::deregisterDef_(
    const OperatorHandle& op,
    const OperatorName& op_name) {
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

namespace {

// Maps OperatorName to (python module name, description) tuple.
using PythonModuleMapType = std::unordered_map<at::OperatorName, std::pair<const char*, const char*>>;
PythonModuleMapType& pythonModulesSingleton() {
  static PythonModuleMapType _data;
  return _data;
}

}

std::optional<std::pair<const char*, const char*>> Dispatcher::getPyStub(OperatorName op_name) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto found = pythonModulesSingleton().find(op_name);
  if (found == pythonModulesSingleton().end()) {
    return std::nullopt;
  }
  return found->second;
}

RegistrationHandleRAII Dispatcher::registerPythonModule(
  const OperatorName& op_name,
  const char* pymodule,
  const char* context
) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  // If there are duplicates, we just let it through and warn about it.
  // Throwing an error during static initialization causes a crash that
  // doesn't give any sign of what happened.
  auto found = pythonModulesSingleton().find(op_name);
  if (found != pythonModulesSingleton().end()) {
    TORCH_WARN(
        "Tried to register an python registration stub (pystub) for ", op_name, " ",
        "that specifies the Python module ", pymodule, " "
        "but there already was a pystub that specifies the Python module ",
        found->second.first, ". We will override the existing pystub.");
  }
  pythonModulesSingleton()[op_name] = std::make_pair(pymodule, context);
  return RegistrationHandleRAII([guard = this->guard_, op_name] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    pythonModulesSingleton().erase(op_name);
  });
}

void Dispatcher::throwIfHasPythonModule(OperatorName op_name) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto elt = pythonModulesSingleton().find(op_name);
  if (elt == pythonModulesSingleton().end()) {
    return;
  }
  const char* pymodule = elt->second.first;
  const char* context = elt->second.second;
  auto* interpreter = at::impl::PythonOpRegistrationTrampoline::getInterpreter();
  TORCH_CHECK(
      interpreter != nullptr,
      op_name,
      ": while attempting to run this operator with Meta Tensors: "
      "Either there is no meta kernel for this operator, or it is located "
      "in the python module ", pymodule, " which is not available "
      "because Python isn't available.")
  (*interpreter)->throw_abstract_impl_not_imported_error(toString(op_name), pymodule, context);
}

RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  std::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  auto op = findOrRegisterName_(op_name);

  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();

  return RegistrationHandleRAII([guard = this->guard_, this, op, op_name, dispatch_key, handle] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}

void Dispatcher::deregisterImpl_(const OperatorHandle& op, const OperatorName& op_name, std::optional<DispatchKey> dispatch_key, impl::OperatorEntry::AnnotatedKernelContainerIterator handle) {
  op.operatorDef_->op.deregisterKernel_(*this, dispatch_key, handle);

  TORCH_INTERNAL_ASSERT(op.operator_name() == op_name);

  TORCH_INTERNAL_ASSERT(op.operatorDef_->def_and_impl_count > 0);
  --op.operatorDef_->def_and_impl_count;

  cleanup(op, op_name);
}

RegistrationHandleRAII Dispatcher::registerName(OperatorName op_name) {
  std::lock_guard<std::mutex> lock(guard_->mutex);
  auto op = findOrRegisterName_(op_name);
  ++op.operatorDef_->def_and_impl_count;

  return RegistrationHandleRAII(
      [guard = this->guard_, this, op, op_name] {
        std::lock_guard<std::mutex> lock(guard->mutex);
        if (!guard->alive.load()) {
          return;
        }
        deregisterName_(op, op_name);
      }
  );
}

void Dispatcher::deregisterName_(
    const OperatorHandle& op,
    const OperatorName& op_name) {
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
  std::lock_guard<std::mutex> lock(guard_->mutex);

  auto idx = getDispatchTableIndexForDispatchKey(dispatchKey);
  TORCH_CHECK(idx >= 0 && static_cast<uint64_t>(idx) < backendFallbackKernels_.size(), "idx=", idx);
  TORCH_CHECK(
    !backendFallbackKernels_[idx].kernel.isValid(),
    "Tried to register multiple backend fallbacks for the same dispatch key ", dispatchKey, "; previous registration ",
    backendFallbackKernels_[idx].debug, ", new registration ", debug
  );
  // NB: inferred function schema is always nullptr for fallbacks, as fallbacks
  // cannot be unboxed
  backendFallbackKernels_[idx] = impl::AnnotatedKernel(std::move(kernel), nullptr, std::move(debug));

  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }

  return RegistrationHandleRAII([guard = this->guard_, this, dispatchKey] {
    std::lock_guard<std::mutex> lock(guard->mutex);
    if (!guard->alive.load()) {
      return;
    }
    deregisterFallback_(dispatchKey);
  });
}

void Dispatcher::deregisterFallback_(DispatchKey dispatchKey) {
  auto idx = getDispatchTableIndexForDispatchKey(dispatchKey);
  backendFallbackKernels_[idx] = {};

  for (auto& op : operators_) {
    op.op.updateFallback(*this, dispatchKey);
  }
}


RegistrationHandleRAII Dispatcher::addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener) {
  std::lock_guard<std::mutex> lock(guard_->mutex);

  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    if (iter->def_count > 0) {
      listener->onOperatorRegistered(OperatorHandle(iter));
    }
  }

  auto removeListener = listeners_->addListener(std::move(listener));
  return RegistrationHandleRAII([guard = this->guard_, this, removeListener] {
      std::lock_guard<std::mutex> lock(guard_->mutex);
      if (!guard->alive.load()) {
        return;
      }
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

std::vector<OperatorName> Dispatcher::getRegistrationsForDispatchKey(std::optional<DispatchKey> k) const {
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

int64_t Dispatcher::sequenceNumberForRunningRecordFunction(DispatchKey dispatchKey, DispatchKeySet dispatchKeySet) {
  int64_t seq_num = -1;
  // Setting sequence number in the Autograd case to associate
  // the forward range with the corresponding Autograd's node

  // Note: this records a sequence number for both Autograd keys, and for
  // non-Autograd keys where the dispatchKeySet still contains an autograd key.
  // This means that we might collect the same sequence nubmer two different
  // events if they all occurred above Autograd and still had the Autograd
  // dispatch key in the dispatch key set.
  // However, this usually doesn't happen: normally the first call will
  // go through the call() or callBoxed() path in the dispatcher, while
  // subsequent redispatches go through redispatch() or redispatchBoxed().
  // `call` has profiler instrumentation, whereas `redispatch` doesn't.
  // So usually, we'll collect a sequence number on the first call() if the
  // dispatch keys contain autograd, and not on subsequent redispatches.
  bool dispatchHasAutograd = !(dispatchKeySet & autograd_dispatch_keyset).empty();

  if (dispatchHasAutograd && at::GradMode::is_enabled()) {
    seq_num = at::sequence_number::peek();
  }
  return seq_num;
}

void Dispatcher::runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey, DispatchKeySet dispatchKeySet, c10::ArrayRef<const c10::IValue> args) {
  guard.before(schema_ref, args, sequenceNumberForRunningRecordFunction(dispatchKey, dispatchKeySet));
}

void Dispatcher::runRecordFunction(at::RecordFunction& guard, at::RecordFunction::schema_ref_t schema_ref, DispatchKey dispatchKey, DispatchKeySet dispatchKeySet) {
  // Setting sequence number in the Autograd case to associate
  // the forward range with the corresponding Autograd's node
  guard.before(schema_ref, sequenceNumberForRunningRecordFunction(dispatchKey, dispatchKeySet));
}
#ifdef FBCODE_CAFFE2
bool Dispatcher::profilingOperatorEvents() {
  return TORCH_SDT_IS_ENABLED(operator_start) || TORCH_SDT_IS_ENABLED(operator_end);
}

C10_NOINLINE void Dispatcher::fireOpStartUSDT(at::RecordFunction::schema_ref_t schema_ref) {
  if (TORCH_SDT_IS_ENABLED(operator_start)) {
    TORCH_SDT_WITH_SEMAPHORE(operator_start, schema_ref.get().name().c_str());
  }
}

C10_NOINLINE void Dispatcher::fireOpEndUSDT(at::RecordFunction::schema_ref_t schema_ref) {
  if (TORCH_SDT_IS_ENABLED(operator_end)) {
    TORCH_SDT_WITH_SEMAPHORE(operator_end, schema_ref.get().name().c_str());
  }
}
#endif

}
