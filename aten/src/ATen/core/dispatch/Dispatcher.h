#pragma once

#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <c10/util/Exception.h>
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
 */
class CAFFE2_API Dispatcher final {
private:
  struct OperatorDef final {
    explicit OperatorDef(FunctionSchema&& schema, OperatorOptions&& options)
    : op(std::move(schema), std::move(options)), refcount(0) {}

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
  SchemaRegistrationHandleRAII registerSchema(FunctionSchema schema, OperatorOptions options);

  /**
   * Looks for an operator schema with the given name and overload name
   * and returns it if it is registered.
   * Returns nullopt otherwise.
   */
  c10::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

  /**
   * Register a kernel to the dispatch table for an operator.
   * If dispatch_key is nullopt, then this registers a fallback kernel.
   *
   * @return A RAII object that manages the lifetime of the registration.
   *         Once that object is destructed, the kernel will be deregistered.
   */
  RegistrationHandleRAII registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction kernel);

  /**
   * Register a fallback kernel for an operator.
   * After this, when trying to lookup a kernel for an unknown dispatch key,
   * it will not fail anymore, but return the fallback kernel instead.
   *
   * @return A RAII object that manages the lifetime of the registration.
   *         Once that object is destructed, the kernel will be deregistered.
   */
  RegistrationHandleRAII registerCatchallKernel(const OperatorHandle& op, KernelFunction kernel);

  template<class Return, class... Args>
  Return callUnboxed(const OperatorHandle& op, TensorTypeId dispatchKey, Args... args) const;

  template<class Return, class... Args>
  Return callUnboxedOnly(const OperatorHandle& op, TensorTypeId dispatchKey, Args... args) const;

  void callBoxed(const OperatorHandle& op, Stack* stack) const;

  /**
   * Add a listener that gets called whenever a new op is registered or an existing
   * op is deregistered. Immediately after registering, this listener gets called
   * for all previously registered ops, so it can be used to keep track of ops
   * registered with this dispatcher.
   */
  void addRegistrationListener(std::unique_ptr<OpRegistrationListener> listener);

private:
  Dispatcher();

  OperatorHandle findOrRegisterSchema_(FunctionSchema&& schema, OperatorOptions&& options);

  void deregisterSchema_(const OperatorHandle& op, const OperatorName& op_name);

  std::list<OperatorDef> operators_;
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
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

  const OperatorOptions& options() const {
    return operatorIterator_->op.options();
  }

private:
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
  : operatorIterator_(std::move(operatorIterator)) {}
  friend class Dispatcher;

  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};

class CAFFE2_API SchemaRegistrationHandleRAII final {
public:
  const OperatorHandle& opHandle() const {
    return opHandle_;
  }

private:
  friend class Dispatcher;
  explicit SchemaRegistrationHandleRAII(OperatorHandle opHandle, RegistrationHandleRAII registrationHandle)
    : opHandle_(std::move(opHandle)), registrationHandle_(std::move(registrationHandle)) {}

  OperatorHandle opHandle_;
  RegistrationHandleRAII registrationHandle_;
};

template<class Return, class... Args>
inline Return Dispatcher::callUnboxed(const OperatorHandle& op, TensorTypeId dispatchKey, Args... args) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  // TODO Remove dispatchKey argument and instead infer dispatchKey from args...
  return op.operatorIterator_->op.callUnboxed<Return, Args...>(std::move(dispatchKey), std::forward<Args>(args)...);
}

template<class Return, class... Args>
inline Return Dispatcher::callUnboxedOnly(const OperatorHandle& op, TensorTypeId dispatchKey, Args... args) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  // TODO Remove dispatchKey argument and instead infer dispatchKey from args...
  return op.operatorIterator_->op.callUnboxedOnly<Return, Args...>(std::move(dispatchKey), std::forward<Args>(args)...);
}

inline void Dispatcher::callBoxed(const OperatorHandle& op, Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  return op.operatorIterator_->op.callBoxed(stack);
}

} // namespace c10
