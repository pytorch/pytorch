#pragma once

#include <ATen/FunctionSequenceNumber.h>
#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/LeftRight.h>
#include <mutex>
#include <list>

namespace c10 {

class CAFFE2_API OperatorHandle;

/**
 * Implement this interface and register your instance with the dispatcher
 * to get notified when operators are registered or deregistered with
 * the dispatcher.
 *
 * NB: registration events only occur when a 'def' occurs; we don't trigger
 * on 'impl' or 'fallback' calls.
 */
class CAFFE2_API OpRegistrationListener {
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
class CAFFE2_API Dispatcher final {
private:
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

public:
  ~Dispatcher();

  // Implementation note: this class abstracts over the fact that we have per-operator
  // dispatch tables.  This could be easily adjusted to have a single global hash
  // table.

  static Dispatcher& singleton();

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
  Return callUnboxed(const OperatorHandle& op, Args... args) const;

  // Like callUnboxed, but override the default DispatchKey calculation code,
  // instead dispatching straight to the provided DispatchKey
  template<class Return, class... Args>
  Return callUnboxedWithDispatchKey(const OperatorHandle& op, DispatchKey dispatchKey, Args... args) const;

  // Like callUnboxed, but intended for use in a redispatch: you are currently
  // in some currentDispatchKey, you have finished processing the key and
  // you now want to redispatch to the next dispatch key in the chain.
  // This will mask out the current key and all previous keys from the
  // eligible set, and redo the calculation.
  template<class Return, class... Args>
  Return callUnboxedRedispatch(const OperatorHandle& op, DispatchKey currentDispatchKey, Args... args) const;

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
  RegistrationHandleRAII registerImpl(OperatorName op_name, c10::optional<DispatchKey> dispatch_key, KernelFunction kernel, std::unique_ptr<FunctionSchema> inferred_function_schema, std::string debug);

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

  // This function is a temporary hack that allows generated_unboxing_wrappers.cpp to register its codegen'ed
  // unboxing wrapper for aten operators. We still need those for some operators because not all work
  // with the templated unboxing logic yet.
  // TODO Delete setBoxedKernelFor_ once all operators work with the templated boxing logic
  void setManuallyBoxedKernelFor_(const OperatorHandle& op, KernelFunction::InternalBoxedKernelFunction* func);

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

private:
  Dispatcher();

  OperatorHandle findOrRegisterSchema_(FunctionSchema&& schema);
  OperatorHandle findOrRegisterName_(const OperatorName& op_name);

  void deregisterDef_(const OperatorHandle& op, const OperatorName& op_name);
  void deregisterImpl_(
    const OperatorHandle& op,
    const OperatorName& op_name,
    c10::optional<DispatchKey> dispatch_key,
    std::list<impl::OperatorEntry::KernelEntry>::iterator kernel_handle);
  void deregisterFallback_(DispatchKey dispatchKey);
  void deregisterLibrary_(const std::string& ns);
  void cleanup(const OperatorHandle& op, const OperatorName& op_name);
  void checkSchemaCompatibility(const OperatorHandle& op, const FunctionSchema& schema, const std::string& debug);

  [[noreturn]] static void reportError(const DispatchTable& dispatchTable, DispatchKey dispatchKey);

  const KernelFunction& dispatch_(const DispatchTable& dispatchTable, DispatchKey dispatch_key) const;

  std::list<OperatorDef> operators_;
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
  // Map from namespace to debug string (saying, e.g., where the library was defined)
  ska::flat_hash_map<std::string, std::string> libraries_;
  impl::KernelFunctionTable backendFallbackKernels_;
  // Set of backends which have specified they do NOT want fallthrough behavior
  // (we store the inverse because it avoids a negation when we use this for
  // masking)
  DispatchKeySet backendsWithoutFallthrough_;
  std::unique_ptr<detail::RegistrationListenerList> listeners_;
  std::mutex mutex_;
};

/**
 * This is a handle to an operator schema registered with the dispatcher.
 * This handle can be used to register kernels with the dispatcher or
 * to lookup a kernel for a certain set of arguments.
 */
class CAFFE2_API OperatorHandle final {
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

  void checkInvariants() const {
    return operatorIterator_->op.checkInvariants();
  }

  template<class Return, class... Args>
  Return callUnboxed(Args... args) const {
    return c10::Dispatcher::singleton().callUnboxed<Return, Args...>(*this, std::forward<Args>(args)...);
  }

  template<class Return, class... Args>
  Return callUnboxedWithDispatchKey(DispatchKey dispatchKey, Args... args) const {
    return c10::Dispatcher::singleton().callUnboxedWithDispatchKey<Return, Args...>(*this, dispatchKey, std::forward<Args>(args)...);
  }

  void callBoxed(Stack* stack) const {
    c10::Dispatcher::singleton().callBoxed(*this, stack);
  }

private:
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : operatorIterator_(std::move(operatorIterator)) {}
  friend class Dispatcher;

  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};

namespace detail {
template<class... Args> inline void unused_arg_(const Args&...) {}

template <typename T, std::enable_if_t<c10::impl::not_ok_to_box<T>::value>* = nullptr>
inline void push_ivalue_copy(std::vector<c10::IValue>& stack, const T& v) {}
template <typename T, std::enable_if_t<!c10::impl::not_ok_to_box<T>::value>* = nullptr>
inline void push_ivalue_copy(std::vector<c10::IValue>& stack, const T& v) {
  torch::jit::push(stack, v);
}

template<typename Item>
void convert_to_ivalue_vector(std::vector<c10::IValue>& stack, const Item& item) {
  push_ivalue_copy(stack, item);
}
template<typename Item, typename... Rest>
void convert_to_ivalue_vector(std::vector<c10::IValue>& stack, const Item& item, Rest... other_items) {
  push_ivalue_copy(stack, item);
  convert_to_ivalue_vector(stack, other_items...);
}
void CAFFE2_API convert_to_ivalue_vector(std::vector<c10::IValue>& stack);
}

template<class Return, class... Args>
inline Return Dispatcher::callUnboxedWithDispatchKey(const OperatorHandle& op, DispatchKey dispatchKey, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  const auto& dispatchTable = op.operatorIterator_->op.dispatch_table();
  const KernelFunction& kernel = dispatch_(dispatchTable, dispatchKey);

  // Check if we need to run callbacks registered with RecordFunction
  // If true and callbacks need inputs, we box the arguments and pass
  // them into the callbacks and also into the kernel call
  at::RecordFunction guard(at::RecordScope::FUNCTION);
  if (guard.active_) {
    // Stack-based record function with scope lifetime can be markee as 'current'
    // thread local record function; used to track nested recorded scopes
    guard._setCurrent();
    if (guard.needs_inputs_) {
      std::vector<c10::IValue> stack;
      detail::convert_to_ivalue_vector(stack, args...);

      guard._before(op.schema().name(), stack, at::FunctionSequenceNumber::peek());

      // if we could convert all the arguments, also pass the stack into the kernel call
      if (stack.size() == sizeof...(args)) {
        return kernel.template callUnboxedWithStack<Return, Args...>(op, stack, std::forward<Args>(args)...);
      }
    } else {
      guard._before(op.schema().name(), at::FunctionSequenceNumber::peek());
    }
  }
  return kernel.template callUnboxed<Return, Args...>(op, std::forward<Args>(args)...);
}

template<class Return, class... Args>
inline Return Dispatcher::callUnboxed(const OperatorHandle& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  const auto& dispatchTable = op.operatorIterator_->op.dispatch_table();
  auto dispatchKey = dispatchTable.dispatchKeyExtractor().getDispatchKeyUnboxed<Args...>(backendsWithoutFallthrough_, DispatchKeySet::FULL, args...);
  return callUnboxedWithDispatchKey<Return, Args...>(op, dispatchKey, args...);
}

template<class Return, class... Args>
inline Return Dispatcher::callUnboxedRedispatch(const OperatorHandle& op, DispatchKey currentDispatchKey, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  const auto& dispatchTable = op.operatorIterator_->op.dispatch_table();
  auto dispatchKey = dispatchTable.dispatchKeyExtractor().getDispatchKeyUnboxed<Args...>(
    backendsWithoutFallthrough_,
    DispatchKeySet(DispatchKeySet::FULL_AFTER, currentDispatchKey),
    args...);
  const KernelFunction& kernel = dispatch_(dispatchTable, dispatchKey);

  // Note: not attempting to use RECORD_FUNCTION to avoid double logging
  return kernel.template callUnboxed<Return, Args...>(op, std::forward<Args>(args)...);
}

inline void Dispatcher::callBoxed(const OperatorHandle& op, Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const auto& dispatchTable = op.operatorIterator_->op.dispatch_table();
  auto dispatchKey = dispatchTable.dispatchKeyExtractor().getDispatchKeyBoxed(backendsWithoutFallthrough_, stack);
  const KernelFunction& kernel = dispatch_(dispatchTable, dispatchKey);

  // using already existing stack to record function execution in observers
  RECORD_FUNCTION(op.schema().name(), *stack, at::FunctionSequenceNumber::peek());

  kernel.callBoxed(op, stack);
}

inline const KernelFunction& Dispatcher::dispatch_(const DispatchTable& dispatchTable, DispatchKey dispatchKey) const {
  const KernelFunction* backendKernel = dispatchTable.lookup(dispatchKey);

  if (nullptr != backendKernel) {
    return *backendKernel;
  }

  const auto& backendFallbackKernel = backendFallbackKernels_[dispatchKey];
  if (backendFallbackKernel.isValid()) {
    return backendFallbackKernel;
  }

  const KernelFunction* catchallKernel = dispatchTable.lookupCatchallKernel();
  if (C10_LIKELY(nullptr != catchallKernel)) {
    return *catchallKernel;
  }

  reportError(dispatchTable, dispatchKey);
}

} // namespace c10
