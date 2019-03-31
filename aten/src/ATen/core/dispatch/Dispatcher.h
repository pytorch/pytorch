#pragma once

#include <ATen/core/dispatch/DispatchTable.h>
#include <c10/util/Exception.h>
#include <mutex>
#include <list>

namespace c10 {

class CAFFE2_API OperatorHandle;

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
  OpKernel(OpKernel&&) = default;
  OpKernel& operator=(OpKernel&&) = default;
  OpKernel(const OpKernel&) = delete;
  OpKernel& operator=(const OpKernel&) = delete;

  /**
   * Call the operator kernel with the given arguments.
   */
  void call(Stack* stack) const {
    return (*kernel_)(stack, cache_.get());
  }

private:
  explicit OpKernel(KernelFunction* kernel, const KernelCacheCreatorFunction& cache_creator)
  : kernel_(kernel), cache_(cache_creator()) {}
  friend class Dispatcher;

  KernelFunction* kernel_;
  std::unique_ptr<c10::KernelCache> cache_;
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

/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 */
class CAFFE2_API Dispatcher final {
private:
  struct OperatorDef final {
    explicit OperatorDef(FunctionSchema&& schema_)
    : dispatchTable(schema_)
    , schema(std::move(schema_))
    , refcount(0) {}

    DispatchTable dispatchTable;
    FunctionSchema schema;
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
   * Register a new operator schema. The handle returned can be used to register
   * kernels to this operator or to call it.
   *
   * If a schema with the same operator name and overload name already exists,
   * this function will check that both schemas are exactly identical and then
   * return the existing schema.
   *
   * Each call to registerSchema() should have a corresponding call to
   * deregisterSchema(), even if multiple calls register (or deregister)
   * schemas with the same operator name and overload name.
   */
  OperatorHandle registerSchema(FunctionSchema schema);

  /**
   * Remove an operator from the dispatcher. Make sure you removed
   * all kernels for this operator before calling this.
   *
   * If a schema was registered multiple times (see above how registerSchema()
   * handles registering schemas that already exist), it must be deregistered
   * the exact same number of times before it is actually deregistered.
   * That is, each call to registerSchema() should have a corresponding call
   * to deregisterSchema().
   */
  void deregisterSchema(const OperatorHandle& op);

  /**
   * Looks for an operator schema with the given name and overload name
   * and returns it if it is registered.
   * Returns nullopt otherwise.
   */
  c10::optional<OperatorHandle> findSchema(const char* operator_name, const char* overload_name);

  /**
   * Register a kernel to the dispatch table for an operator.
   * If dispatch_key is nullopt, then this registers a fallback kernel.
   */
  void registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func);

  /**
   * Remove a kernel from the dispatch table for an operator.
   * If dispatch_key is none, then this deregisters the fallback kernel.
   * See documentation for registerKernel() for details.
   */
  void deregisterKernel(const OperatorHandle& op, TensorTypeId dispatch_key);

  /**
   * Register a fallback kernel for an operator.
   * After this, when trying to lookup a kernel for an unknown dispatch key,
   * it will not fail anymore, but return the fallback kernel instead.
   */
  void registerFallbackKernel(const OperatorHandle& op, KernelFunction* kernel_func, KernelCacheCreatorFunction cache_creator_func);

  /**
   * Remove the fallback kernel for an operator.
   * After this, if trying to lookup a kernel for an unknown dispatch key,
   * the lookup will fail.
   */
  void deregisterFallbackKernel(const OperatorHandle& op);

  /**
   * Perform a dynamic dispatch and get the kernel for an operator.
   */
  OpKernel lookup(const OperatorHandle& op, const Stack* stack) const;

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

  std::list<OperatorDef> operators_;
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
  OperatorHandle(OperatorHandle&&) = default;
  OperatorHandle& operator=(OperatorHandle&&) = default;
  OperatorHandle(const OperatorHandle&) = default;
  OperatorHandle& operator=(const OperatorHandle&) = default;

  const FunctionSchema& schema() const {
    return operatorDefIterator_->schema;
  }

private:
  explicit OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorDefIterator)
  : operatorDefIterator_(std::move(operatorDefIterator)) {}
  friend class Dispatcher;

  std::list<Dispatcher::OperatorDef>::iterator operatorDefIterator_;
};


inline OpKernel Dispatcher::lookup(const OperatorHandle& op, const Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const DispatchTableEntry& kernel = op.operatorDefIterator_->dispatchTable.lookup(stack);
  return OpKernel(kernel.kernel_func, kernel.cache_creator_func);
}

} // namespace c10
