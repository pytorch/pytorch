#pragma once

#include <ATen/core/dispatch/OperatorEntry.h>

namespace c10 {

inline void OpKernel::call(Stack* stack) const {
  // TODO Make boxed kernels mandatory and remove this check
  TORCH_CHECK(nullptr != kernel_, "Tried to call OpKernel::call() for a kernel that doesn't have an boxed version.");

  return (*kernel_)(stack, cache_.get());
}

template<class Result, class... Args>
inline Result OpKernel::callUnboxed(Args... args) const {
  // TODO Should we box and call the boxed kernel instead of failing?
  TORCH_CHECK(nullptr != unboxed_kernel_, "Tried to call OpKernel::callUnboxed() for a kernel that doesn't have an unboxed version.");

  using OpSignature = Result (c10::KernelCache*, Args...);
  OpSignature* kernel = reinterpret_cast<OpSignature*>(unboxed_kernel_);
  return (*kernel)(cache_.get(), std::forward<Args>(args)...);
}

inline OpKernel::OpKernel(KernelFunction* kernel, const KernelCacheCreatorFunction& cache_creator, void* unboxed_kernel)
: kernel_(kernel), cache_(cache_creator()), unboxed_kernel_(unboxed_kernel) {}

struct Dispatcher::OperatorDef final {
  explicit OperatorDef(FunctionSchema&& schema, OperatorOptions&& options)
  : op(std::move(schema), std::move(options)), refcount(0) {}

  impl::OperatorEntry op;
  size_t refcount;
};

inline const FunctionSchema& OperatorHandle::schema() const {
  return operatorIterator_->op.schema();
}

inline const OperatorOptions& OperatorHandle::options() const {
  return operatorIterator_->op.options();
}

inline OperatorHandle::OperatorHandle(std::list<Dispatcher::OperatorDef>::iterator operatorIterator)
: operatorIterator_(std::move(operatorIterator)) {}

inline const OperatorHandle& SchemaRegistrationHandleRAII::opHandle() const {
  return opHandle_;
}

inline SchemaRegistrationHandleRAII::SchemaRegistrationHandleRAII(OperatorHandle opHandle, RegistrationHandleRAII registrationHandle)
  : opHandle_(std::move(opHandle)), registrationHandle_(std::move(registrationHandle)) {}

inline OpKernel Dispatcher::lookup(const OperatorHandle& op, const Stack* stack) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const DispatchTableEntry& kernel = op.operatorIterator_->op.lookupKernel(stack);
  return OpKernel(kernel.kernel_func, kernel.cache_creator_func, kernel.unboxed_kernel_func);
}

inline OpKernel Dispatcher::lookup(const OperatorHandle& op, TensorTypeId dispatchKey) const {
  // note: this doesn't need the mutex because write operations on the list keep iterators intact.
  const DispatchTableEntry& kernel = op.operatorIterator_->op.lookupKernel(dispatchKey);
  return OpKernel(kernel.kernel_func, kernel.cache_creator_func, kernel.unboxed_kernel_func);
}

template<class Result, class... Args>
inline Result Dispatcher::callUnboxedAutogradKernel(const OperatorHandle& op, Args... args) const {
  void* unboxed_autograd_kernel = op.operatorIterator_->op.lookupUnboxedAutogradKernel();
  TORCH_CHECK(nullptr != unboxed_autograd_kernel, "Tried to call Dispatcher::callUnboxedAutogradKernel() for an operator that doesn't have an autograd kernel.");

  using OpSignature = Result (Args...);
  OpSignature* kernel = reinterpret_cast<OpSignature*>(unboxed_autograd_kernel);
  return (*kernel)(std::forward<Args>(args)...);
}

}
