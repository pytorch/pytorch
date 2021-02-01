#pragma once

#include <ATen/SequenceNumber.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/boxing/impl/boxing.h>
#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/LeftRight.h>
#include <mutex>
#include <list>

#include <ATen/core/grad_mode.h>

namespace c10 {

class TORCH_API OperatorHandle;
template<class FuncType> class TypedOperatorHandle;

/**
 * Implement this interface and register your instance with the dispatcher
 * to get notified when operators are registered or deregistered with
 * the dispatcher.
 *
 * NB: registration events only occur when a 'def' occurs; we don't trigger
 * on 'impl' or 'fallback' calls.
 */
class TORCH_API OpRegistrationListener {
public:
  virtual ~OpRegistrationListener();

  virtual void onOperatorRegistered(const OperatorHandle& op) = 0;
  virtual void onOperatorDeregistered(const OperatorHandle& op) = 0;
};

namespace detail {
class RegistrationListenerList;
}
class SchemaRegistrationHandleRAII;

/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 * Most end users shouldn't use this directly; if you're trying to register
 * ops look in op_registration
 */
class TORCH_API Dispatcher final {
private:
  // For direct access to backend fallback information
  friend class impl::OperatorEntry;

  struct OperatorDef final {
    explicit OperatorDef(OperatorName&& op_name)
    : op(std::move(op_name)) {}

    impl::OperatorEntry op;

    // These refer to the number of outstanding RegistrationHandleRAII
    // for this operator.  def_count reflects only def() registrations
    // (in the new world, this should only ever be 1, but old style
    // registrations may register the schema multiple times, which
    // will increase this count).  def_and_impl_count reflects the number
    // of combined def() and impl() registrations.  When the last def() gets
    // unregistered, we must immediately call the Deregistered listeners, but we
    // must not actually delete the handle as there are other outstanding RAII
    // destructors which will try to destruct and they had better still have a
    // working operator handle in this case
    size_t def_count = 0;
    size_t def_and_impl_count = 0;
  };
  friend class OperatorHandle;
  template<class> friend class TypedOperatorHandle;

public:
  ~Dispatcher();

  // Implementation note: this class abstracts over the fact that we have per-operator
  // dispatch tables.  This could be easily adjusted to have a single global hash
  // table.
  static Dispatcher& realSingleton();

  static Dispatcher& singleton() {
    // Implemented inline so that steady-state code needn't incur
    // function-call overhead. We can't just inline `realSingleton`
    // because the function-local static would get duplicated across
    // all DSOs that include & use this header, leading to multiple
    // singleton instances.
    static Dispatcher& s = realSingleton();
    return s;
  }

  // ------------------------------------------------------------------------
  //
  // Accessing operators by schema
  //
  // ------------------------------------------------------------------------

  /**
   * Looks for an operator schema with the given name and overload name
   * and returns it if it is registered WITH A SCHEMA.
   * Returns nullopt otherwise.
   */
  c10::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

  /**
   * Variant of findSchema that results in less code generated at the call site.
   * It (1) takes const char* pointer rather than OperatorName (so we skip
   * generating std::string constructor calls at the call site), and (2)
   * it raises an exception if the operator is not found (so we skip
   * generating exception raising code at the call site)
   *
   * Irritatingly, we still have to generate the handful of instructions
   * for dealing with an exception being thrown during static initialization
   * (e.g. __cxa_guard_abort).  If we could annotate this method noexcept we
   * could avoid this code too, but as the name of the function suggests,
   * it does throw exceptions.
   */
  OperatorHandle findSchemaOrThrow(const char* name, const char* overload_name);

  // Like findSchema, but also returns OperatorHandle even if there is no schema
  c10::optional<OperatorHandle> findOp(const OperatorName& operator_name);

  // ------------------------------------------------------------------------
  //
  // Invoking operators
  //
  // ------------------------------------------------------------------------

  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return (Args...)>& op, Args... args) const;

  // Like call, but override the default DispatchKey calculation code,
  // instead dispatching straight to the provided DispatchKey
  template<class Return, class... Args>
  Return callWithDispatchKey(const TypedOperatorHandle<Return (Args...)>& op, DispatchKey dispatchKey, Args... args) const;

  template<class Return, class... Args>
  static Return callWithDispatchKeySlowPath(const TypedOperatorHandle<Return (Args...)>& op, bool pre_sampled, DispatchKey dispatchKey, const KernelFunction& kernel, Args... args);

  // Like call, but intended for use in a redispatch: you are currently
  // in some currentDispatchKey, you have finished processing the key and
  // you now want to redispatch to the next dispatch key in the chain.
  // This will mask out the current key *and all previous keys* from the
  // eligible set, and reinvoke the dispatcher.
  template<class Return, class... Args>
  Return redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKey currentDispatchKey, Args... args) const;

  // Invoke an operator via the boxed calling convention using an IValue stack
  void callBoxed(const OperatorHandle& op, Stack* stack) const;

  // ------------------------------------------------------------------------
  //
  // Performing registrations (NON user public; use op_registration)
  //
  // ------------------------------------------------------------------------

  /**
   * Register a new operator schema.
   *
   * If a schema with the same operator name and overload name already exists,
   * this function will check that both schemas are exactly identical.
   */
  RegistrationHandleRAII registerDef(FunctionSchema schema, std::string debug);

  /**
   * Register a kernel to the dispatch table for an operator.
   * If dispatch_key is nullopt, then this registers a fallback kernel.
   *
   * @return A RAII object that manages the lifetime of the registration.
   *         Once that object is destructed, the kernel will be deregistered.
   */
  // NB: steals the inferred function schema, as we may need to hold on to
  // it for a bit until the real schema turns up
  RegistrationHandleRAII registerImpl(OperatorName op_name, c10::optional<DispatchKey> dispatch_key, KernelFunction kernel, c10::optional<impl::CppSignature> cpp_signature, std::unique_ptr<FunctionSchema> inferred_function_schema, std::string debug);

  /**
   * Register a new operator by name.
   */
  RegistrationHandleRAII registerName(OperatorName op_name);

  /**
   * Register a fallback kernel for a backend.
   * If an operator is called but there is no concrete kernel for the dispatch
   * key of the given operator arguments, it will check if there is such a
   * fallback kernel for the given dispatch key and, if yes, call that one.
   */
  RegistrationHandleRAII registerFallback(DispatchKey dispatch_key, KernelFunction kernel, std::string debug);

  /**
   * Use to register whenever we had a TORCH_LIBRARY declaration in the frontend
   * API.  These invocations are only permitted once per program, so we raise
   * an error if this is called again for the same namespace.
   */
  RegistrationHandleRAII registerLibrary(std::string ns, std::string debug);

  // ------------------------------------------------------------------------
  //
  // Listeners on registrations
  //
  // ------------------------------------------------------------------------

  /**
   * Add a listener that gets called whenever a new op is registered or an existing
   * op is deregistered. Immediately after registering, this listener gets called
   * for all previously registered ops, so it can be used to keep track of ops
   * registered with this dispatcher.
   */
  RegistrationHandleRAII addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener);

  void checkInvariants() const;

  /* Check if operator calls with a given dispatch key
   * need to be observed with RecordFunction.
   */
  inline static bool shouldRecord(DispatchKey dispatch_key) {
    return dispatch_key != DispatchKey::BackendSelect;
  }

  //
  // ------------------------------------------------------------------------
  //
  // Assertions
  //
  // ------------------------------------------------------------------------

  /**
   * For testing purposes.
   * Returns a list of all operators that were created through calls to registerImpl(),
   * without any corresponding calls to registerDef(). After static initialization
   * is done this is almost certainly a bug, as the created OperatorHandle won't have
   * any schema associated with it and users calling the op through the dispatcher
   * won't be able to access it
   *
   * Note that we cannot enforce this invariant "as we go" during static initialization,
   * due to undefined static initialization order- we have no guarantees over the order
   * in which .def() and .impl() calls are registered in the dispatcher at static
   * initialization time. So this function should only be called after static initialization.
   */
  std::vector<OperatorHandle> findDanglingImpls() const;

private:
  Dispatcher();

  static int64_t sequenceNumberForRunningRecordFunction(DispatchKey dispatchKey);
  static void runRecordFunction(at::RecordFunction& guard, const OperatorHandle& op, DispatchKey dispatchKey);
  static void runRecordFunction(at::RecordFunction& guard, const OperatorHandle& op, DispatchKey dispatchKey, torch::jit::Stack &&stack);
  static void runRecordFunction(at::RecordFunction& guard, const OperatorHandle& op, DispatchKey dispatchKey, const torch::jit::Stack &stack);

  OperatorHandle findOrRegisterSchema_(FunctionSchema&& schema);
  OperatorHandle findOrRegisterName_(const OperatorName& op_name);

  void deregisterDef_(const OperatorHandle& op, const OperatorName& op_name);
  void deregisterImpl_(
    const OperatorHandle& op,
    const OperatorName& op_name,
    c10::optional<DispatchKey> dispatch_key,
    std::list<impl::AnnotatedKernel>::iterator kernel_handle);
  void deregisterName_(const OperatorHandle& op, const OperatorName& op_name);
  void deregisterFallback_(DispatchKey dispatchKey);
  void deregisterLibrary_(const std::string& ns);
  void cleanup(const OperatorHandle& op, const OperatorName& op_name);
  void checkSchemaCompatibility(const OperatorHandle& op, const FunctionSchema& schema, const std::string& debug);

  std::list<OperatorDef> operators_;
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
  // Map from namespace to debug string (saying, e.g., where the library was defined)
  ska::flat_hash_map<std::string, std::string> libraries_;

  std::array<impl::AnnotatedKernel, static_cast<uint8_t>(DispatchKey::NumDispatchKeys)> backendFallbackKernels_;

  std::unique_ptr<detail::RegistrationListenerList> listeners_;
  std::mutex mutex_;
};

/**
 * This is a handle to an operator schema registered with the dispatcher.
 * This handle can be used to register kernels with the dispatcher or
 * to lookup a kernel for a certain set of arguments.
 */
class TORCH_API OperatorHandle {
public:
  OperatorHandle(OperatorHandle&&) noexcept = default;
  OperatorHandle& operator=(OperatorHandle&&) noexcept = default;
  OperatorHandle(const OperatorHandle&) = default;
  OperatorHandle& operator=(const OperatorHandle&) = default;

  const OperatorName& operator_name() const {
    return operatorIterator_->op.operator_name();
  }

  bool hasSchema() const {
    return operatorIterator_->op.hasSchema();
  }

  const FunctionSchema& schema() const {
    return operatorIterator_->op.schema();
  }

  const std::string& debug() const {
    return operatorIterator_->op.debug();
  }

  std::string dumpState() const {
    return operatorIterator_->op.dumpState();
  }

  std::string dumpComputedTable() const {
    return operatorIterator_->op.dumpComputedTable();
  }

  void checkInvariants() const {
    return operatorIterator_->op.checkInvariants();
  }

  template<class FuncType>
  TypedOperatorHandle<FuncType> typed() const {
    // NB: This assert is not 100% sound: you can retrieve a typed() operator
    // handle prior to ANY C++ signature being registered on the operator
    // and the check will say everything is OK (at which point you can then
    // smuggle in a kernel that is typed incorrectly).  For everything
    // in core library this won't happen, because all the static registrations
    // will be done by the time a typed() handle is acquired.
#if !defined C10_MOBILE
    operatorIterator_->op.assertSignatureIsCorrect<FuncType>();
#endif
    return TypedOperatorHandle<FuncType>(operatorIterator_);
  }

  void callBoxed(Stack* stack) const {
    c10::Dispatcher::singleton().callBoxed(*this, stack);
  }

private:
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : operatorIterator_(std::move(operatorIterator)) {}
  friend class Dispatcher;
  template<class> friend class TypedOperatorHandle;

  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};

/**
 * This is a handle to an operator schema registered with the dispatcher.
 * It holds the same information as an OperatorHandle, but it is templated
 * on the operator arguments and allows calling the operator in an
 * unboxed way.
 */
template<class FuncType>
class TypedOperatorHandle final {
  static_assert(guts::false_t<FuncType>(), "FuncType in OperatorHandle::typed<FuncType> was not a valid function type");
};
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {
public:
  TypedOperatorHandle(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle& operator=(TypedOperatorHandle&&) noexcept = default;
  TypedOperatorHandle(const TypedOperatorHandle&) = default;
  TypedOperatorHandle& operator=(const TypedOperatorHandle&) = default;

  Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
  }

  Return callWithDispatchKey(DispatchKey dispatchKey, Args... args) const {
    return c10::Dispatcher::singleton().callWithDispatchKey<Return, Args...>(*this, dispatchKey, std::forward<Args>(args)...);
  }

private:
  explicit TypedOperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : OperatorHandle(std::move(operatorIterator)) {}
  friend class OperatorHandle;
};

namespace detail {
template<class... Args> inline void unused_arg_(const Args&...) {}
}

template<class Return, class... Args>
inline Return Dispatcher::callWithDispatchKey(const TypedOperatorHandle<Return(Args...)>& op, DispatchKey dispatchKey, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  // No alias dispatch key is allowed at runtime.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!c10::isAliasDispatchKey(dispatchKey));
  const KernelFunction& kernel = op.operatorIterator_->op.lookup(dispatchKey);

#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  // By default, when there're no high-frequency or non-sampled callbacks,
  // RecordFunction is pre-sampled as a perf optimization;
  // shouldRunRecordFunction checks whether RecordFunction should be executed,
  // and sets pre_sampled boolean argument value to whether pre-sampling was used -
  // this boolean is passed into RecordFunction to adjust the sampling rates of
  // the callbacks
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    return callWithDispatchKeySlowPath<Return, Args...>(op, pre_sampled, dispatchKey, kernel, std::forward<Args>(args)...);
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING
  return kernel.template call<Return, Args...>(op, std::forward<Args>(args)...);
}

template<class Return, class... Args>
inline Return Dispatcher::callWithDispatchKeySlowPath(const TypedOperatorHandle<Return(Args...)>& op, bool pre_sampled, DispatchKey dispatchKey, const KernelFunction& kernel, Args... args) {
    // Check if we need to run callbacks registered with RecordFunction
    // If true and callbacks need inputs, we box the arguments and pass
    // them into the callbacks and also into the kernel call

    // Note: for perf reasons we wouldn't want to pass arguments into
    // the function call or prematurely box them
  at::RecordFunction guard(at::RecordScope::FUNCTION, pre_sampled);
  if (C10_UNLIKELY(guard.isActive())) {
    if (shouldRecord(dispatchKey) && op.operatorIterator_->op.isObserved()) {
      if (guard.needsInputs()) {
        runRecordFunction(guard, op, dispatchKey, impl::boxArgs(args...));
      } else {
        runRecordFunction(guard, op, dispatchKey);
      }
    }
  }
  // keeping the guard alive while executing the kernel
  return kernel.template call<Return, Args...>(op, std::forward<Args>(args)...);
}

template<class Return, class... Args>
inline Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKey = op.operatorIterator_->op.dispatchKeyExtractor()
    .template getDispatchKeyUnboxed<Args...>(
      DispatchKeySet::FULL,
      args...
    );
  return callWithDispatchKey<Return, Args...>(op, dispatchKey, args...);
}

template<class Return, class... Args>
inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKey currentDispatchKey, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  auto dispatchKey = op.operatorIterator_->op.dispatchKeyExtractor()
    .template getDispatchKeyUnboxed<Args...>(
      DispatchKeySet(DispatchKeySet::FULL_AFTER, currentDispatchKey),
      args...
    );
  // do not use RecordFunction on redispatch
  const KernelFunction& kernel = op.operatorIterator_->op.lookup(dispatchKey);
  return kernel.template call<Return, Args...>(op, std::forward<Args>(args)...);
}

inline void Dispatcher::callBoxed(const OperatorHandle& op, Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const auto& entry = op.operatorIterator_->op;
  auto dispatchKey = entry.dispatchKeyExtractor().getDispatchKeyBoxed(stack);
  const auto& kernel = entry.lookup(dispatchKey);

#ifndef PYTORCH_DISABLE_PER_OP_PROFILING
  bool pre_sampled = false;
  if (C10_UNLIKELY(at::shouldRunRecordFunction(&pre_sampled))) {
    // using already existing stack to record function execution in observers
    at::RecordFunction guard(at::RecordScope::FUNCTION, pre_sampled);
    if (C10_UNLIKELY(guard.isActive())) {
      if (shouldRecord(dispatchKey) && entry.isObserved()) {
        if (guard.needsInputs()) {
          runRecordFunction(guard, op, dispatchKey, *stack);
        } else {
          runRecordFunction(guard, op, dispatchKey);
        }
      }
    }
    // keeping the guard alive while executing the kernel
    kernel.callBoxed(op, stack);
    return;
  }
#endif  // PYTORCH_DISABLE_PER_OP_PROFILING
  kernel.callBoxed(op, stack);
}

} // namespace c10
