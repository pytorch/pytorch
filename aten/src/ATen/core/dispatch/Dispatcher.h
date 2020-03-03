#pragma once

#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
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
    explicit OperatorDef(FunctionSchema&& schema)
    : op(std::move(schema)), refcount(0) {}

    impl::OperatorEntry op;
    size_t refcount;
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
   * and returns it if it is registered.
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
   *
   * @return An OperatorHandle for the registered schema which can be used to
   *         register kernels for the operator and a RegistrationHandleRAII RAII
   *         object that manages the lifetime of the registration. Once that
   *         object is destructed, the kernel will be deregistered.
   */
  std::pair<RegistrationHandleRAII, OperatorHandle> registerSchema(FunctionSchema schema);

  /**
   * Register a kernel to the dispatch table for an operator.
   * If dispatch_key is nullopt, then this registers a fallback kernel.
   *
   * @return A RAII object that manages the lifetime of the registration.
   *         Once that object is destructed, the kernel will be deregistered.
   */
  RegistrationHandleRAII registerKernel(const OperatorHandle& op, c10::optional<DispatchKey> dispatch_key, KernelFunction kernel);

  /**
   * Register a fallback kernel for a backend.
   * If an operator is called but there is no concrete kernel for the dispatch
   * key of the given operator arguments, it will check if there is such a
   * fallback kernel for the given dispatch key and, if yes, call that one.
   */
  RegistrationHandleRAII registerBackendFallbackKernel(DispatchKey dispatch_key, KernelFunction kernel);

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
  void addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener);

private:
  Dispatcher();

  OperatorHandle findOrRegisterSchema_(FunctionSchema&& schema);

  void deregisterSchema_(const OperatorHandle& op, const OperatorName& op_name);
  void deregisterBackendFallbackKernel_(DispatchKey dispatchKey);
  [[noreturn]] static void reportError(const DispatchTable& dispatchTable, DispatchKey dispatchKey);

  const KernelFunction& dispatch_(const DispatchTable& dispatchTable, DispatchKey dispatch_key) const;

  std::list<OperatorDef> operators_;
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
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

  const FunctionSchema& schema() const {
    return operatorIterator_->op.schema();
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
}

template<class Return, class... Args>
inline Return Dispatcher::callUnboxedWithDispatchKey(const OperatorHandle& op, DispatchKey dispatchKey, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  const auto& dispatchTable = op.operatorIterator_->op.dispatch_table();
  const KernelFunction& kernel = dispatch_(dispatchTable, dispatchKey);
  return kernel.template callUnboxed<Return, Args...>(op, std::forward<Args>(args)...);
}

template<class Return, class... Args>
inline Return Dispatcher::callUnboxed(const OperatorHandle& op, Args... args) const {
  detail::unused_arg_(args...);  // workaround for a false-positive warning about unused parameters in gcc 5
  const auto& dispatchTable = op.operatorIterator_->op.dispatch_table();
  auto dispatchKey = dispatchTable.dispatchKeyExtractor().getDispatchKeyUnboxed<Args...>(backendsWithoutFallthrough_, args...);
  return callUnboxedWithDispatchKey<Return, Args...>(op, dispatchKey, args...);
}

inline void Dispatcher::callBoxed(const OperatorHandle& op, Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const auto& dispatchTable = op.operatorIterator_->op.dispatch_table();
  auto dispatchKey = dispatchTable.dispatchKeyExtractor().getDispatchKeyBoxed(backendsWithoutFallthrough_, stack);
  const KernelFunction& kernel = dispatch_(dispatchTable, dispatchKey);
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
