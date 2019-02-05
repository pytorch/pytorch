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
 * OpKernel is not threadsafe.
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
    if (cache_.get() == nullptr) {
      AT_ASSERT(cache_creator_ != nullptr);
      cache_ = (*cache_creator_)();
    }
    return (*kernel_)(stack, cache_.get());
  }

private:
  explicit OpKernel(KernelFunction* kernel, KernelCacheCreatorFunction* cache_creator)
  : kernel_(kernel), cache_creator_(cache_creator) {}
  friend class Dispatcher;

  KernelFunction* kernel_;

  KernelCacheCreatorFunction* cache_creator_;
  mutable std::unique_ptr<c10::KernelCache> cache_;
};

/**
 * Top-level dispatch interface for dispatching via the dynamic dispatcher.
 */
class CAFFE2_API Dispatcher final {
private:
  struct OperatorDef final {
    explicit OperatorDef(FunctionSchema schema_)
    : dispatchTable(schema_)
    , schema(std::move(schema_)) {}

    DispatchTable dispatchTable;
    FunctionSchema schema;
  };
  friend class OperatorHandle;

public:
  // Implementation note: this class abstracts over the fact that we have per-operator
  // dispatch tables.  This could be easily adjusted to have a single global hash
  // table.

  static Dispatcher& singleton();

  /**
   * Register a new operator schema. The handle returned can be used to register
   * kernels to this operator or to call it.
   */
  OperatorHandle registerSchema(FunctionSchema schema);

  /**
   * Remove an operator from the dispatcher. Make sure you removed
   * all kernels for this operatorbefore calling this.
   */
  void deregisterSchema(const OperatorHandle& op);

  /**
   * Register an operator to the dispatch table for an operator.
   */
  void registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction* cache_creator_func);

  /**
   * Remove an operator from the dispatch table for an operator.
   */
  void deregisterKernel(const OperatorHandle& op, TensorTypeId dispatch_key);

  /**
   * Perform a dynamic dispatch and get the kernel for an operator.
   */
  OpKernel lookup(const OperatorHandle& op, const Stack* stack) const;

private:
  std::list<OperatorDef> operators_;
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



inline OperatorHandle Dispatcher::registerSchema(FunctionSchema schema) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  operators_.emplace_back(std::move(schema));
  return OperatorHandle(--operators_.end());
}

inline void Dispatcher::deregisterSchema(const OperatorHandle& op) {
  // we need a lock to avoid concurrent writes
  std::lock_guard<std::mutex> lock(mutex_);

  if (!op.operatorDefIterator_->dispatchTable.isEmpty()) {
    AT_ERROR("Tried to deregister op schema that still has kernels registered");
  }
  operators_.erase(op.operatorDefIterator_);
}

inline void Dispatcher::registerKernel(const OperatorHandle& op, TensorTypeId dispatch_key, KernelFunction* kernel_func, KernelCacheCreatorFunction* cache_creator_func) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.registerKernel(std::move(dispatch_key), DispatchTableEntry{kernel_func, cache_creator_func});
}

inline void Dispatcher::deregisterKernel(const OperatorHandle& op, TensorTypeId dispatch_key) {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  op.operatorDefIterator_->dispatchTable.deregisterKernel(dispatch_key);
}

inline OpKernel Dispatcher::lookup(const OperatorHandle& op, const Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const DispatchTableEntry& kernel = op.operatorDefIterator_->dispatchTable.lookup(stack);
  return OpKernel(kernel.kernel_func, kernel.cache_creator_func);
}

} // namespace c10
