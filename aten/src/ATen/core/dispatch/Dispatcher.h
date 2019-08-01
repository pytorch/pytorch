#pragma once

#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/KernelFunction.h>
#include <ATen/core/operator_name.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/LeftRight.h>
#include <c10/util/flat_hash_map.h>
#include <c10/core/TensorTypeId.h>
#include <mutex>
#include <list>

namespace c10 {

class CAFFE2_API OperatorHandle;
class FunctionSchema;

/**
 * This class represents an operator kernel, i.e. an operator *after* it was
 * dispatched to a certain device. You can use it to call the kernel.
 *
 * You can keep this OpKernel instance around to avoid future dispatch
 * when you know it'd dispatch to the same kernel anyhow.
 *
 * Also, keeping around the OpKernel instance will keep around a local cache
 * that is used by some kernels to get better performance when they're called
 * multiple times (mostly Caffe2 kernels do that).
 *
 * OpKernel is only threadsafe if the kernel is threadsafe. There are no mutexes
 * protecting the kernel cache, so if the kernel uses the cache and doesn't have
 * mutexes for it, it will likely not be threadsafe.
 */
class CAFFE2_API OpKernel final {
public:
  OpKernel(OpKernel&&) noexcept = default;
  OpKernel& operator=(OpKernel&&) noexcept = default;
  OpKernel(const OpKernel&) = delete;
  OpKernel& operator=(const OpKernel&) = delete;

  /**
   * Call the operator kernel with the given arguments.
   */
  void call(Stack* stack) const;

  template<class Result, class... Args>
  Result callUnboxed(Args... args) const;

private:
  explicit OpKernel(KernelFunction* kernel, const KernelCacheCreatorFunction& cache_creator, void* unboxed_kernel);
  friend class Dispatcher;

  KernelFunction* kernel_; // can be nullptr, not all kernels have this
  std::unique_ptr<c10::KernelCache> cache_;
  void* unboxed_kernel_; // can be nullptr, not all kernels have this
};

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
  struct OperatorDef;
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
  SchemaRegistrationHandleRAII registerSchema(FunctionSchema&& schema, OperatorOptions options);

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
  RegistrationHandleRAII registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func, void* unboxed_kernel_func);

  /**
   * Register a fallback kernel for an operator.
   * After this, when trying to lookup a kernel for an unknown dispatch key,
   * it will not fail anymore, but return the fallback kernel instead.
   *
   * @return A RAII object that manages the lifetime of the registration.
   *         Once that object is destructed, the kernel will be deregistered.
   */
  RegistrationHandleRAII registerCatchallKernel(const OperatorHandle& op, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func, void* unboxed_kernel_func);

  RegistrationHandleRAII registerUnboxedAutogradKernel(const OperatorHandle& op, void* unboxed_autograd_kernel);

  /**
   * Perform a dynamic dispatch and get the kernel for an operator.
   */
  OpKernel lookup(const OperatorHandle& op, const Stack* stack) const;

  /**
   * Perform a dynamic dispatch and get the kernel for an operator.
   */
  OpKernel lookup(const OperatorHandle& op, TensorTypeId dispatchKey) const;

  template<class Result, class... Args>
  Result callUnboxedAutogradKernel(const OperatorHandle& op, Args... args) const;

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

  const FunctionSchema& schema() const;

  const OperatorOptions& options() const;

private:
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator);
  friend class Dispatcher;

  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};

class CAFFE2_API SchemaRegistrationHandleRAII final {
public:
  const OperatorHandle& opHandle() const;

private:
  explicit SchemaRegistrationHandleRAII(OperatorHandle opHandle, RegistrationHandleRAII registrationHandle);
  friend class Dispatcher;

  OperatorHandle opHandle_;
  RegistrationHandleRAII registrationHandle_;
};

} // namespace c10

#include "Dispatcher_inl.h"
