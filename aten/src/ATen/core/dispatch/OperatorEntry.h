#pragma once

#include <ATen/core/dispatch/DispatchTable.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <list>

namespace c10 {
namespace impl {
  class OperatorEntry;
}

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
  void call(Stack* stack) const {
    // TODO Make boxed kernels mandatory and remove this check
    TORCH_CHECK(nullptr != kernel_, "Tried to call OpKernel::call() for a kernel that doesn't have an boxed version.");

    return (*kernel_)(stack, cache_.get());
  }

  template<class Result, class... Args>
  Result callUnboxed(Args... args) const {
    // TODO Should we box and call the boxed kernel instead of failing?
    TORCH_CHECK(nullptr != unboxed_kernel_, "Tried to call OpKernel::callUnboxed() for a kernel that doesn't have an unboxed version.");

    using OpSignature = Result (c10::KernelCache*, Args...);
    OpSignature* kernel = reinterpret_cast<OpSignature*>(unboxed_kernel_);
    return (*kernel)(cache_.get(), std::forward<Args>(args)...);
  }

private:
  explicit OpKernel(KernelFunction* kernel, const KernelCacheCreatorFunction& cache_creator, void* unboxed_kernel)
  : kernel_(kernel), cache_(cache_creator ? cache_creator() : nullptr), unboxed_kernel_(unboxed_kernel) {}
  friend class impl::OperatorEntry;

  // All of these fields may be nullptr, but at least one of
  // kernel_ or unboxed_kernel_ should be non-NULL
  KernelFunction* kernel_;
  std::unique_ptr<c10::KernelCache> cache_;
  void* unboxed_kernel_;
};

namespace impl {

// This is a private class used inside the Dispatcher to represent an operator
// and its dispatch table. This is not part of the public API.
class OperatorEntry final {
public:
  explicit OperatorEntry(FunctionSchema&& schema, OperatorOptions&& options);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  const FunctionSchema& schema() const {
    return schema_;
  }

  OpKernel lookupKernel(const Stack* stack) const {
    return dispatchTable_.read([&] (const DispatchTable& dispatchTable) {
      const DispatchTableEntry& kernel = dispatchTable.lookup(stack);
      return OpKernel(kernel.kernel_func, kernel.cache_creator_func, kernel.unboxed_kernel_func);
    });
  }

  OpKernel lookupKernel(TensorTypeId dispatchKey) const {
    return dispatchTable_.read([&] (const DispatchTable& dispatchTable) {
      const DispatchTableEntry& kernel = dispatchTable.lookup(dispatchKey);
      return OpKernel(kernel.kernel_func, kernel.cache_creator_func, kernel.unboxed_kernel_func);
    });
  }

  void* lookupUnboxedAutogradKernel() const {
    return currentUnboxedAutogradKernel_;
  }

  void prepareForDeregistration();

  RegistrationHandleRAII registerKernel(TensorTypeId dispatch_key, DispatchTableEntry kernel);
  RegistrationHandleRAII registerCatchallKernel(DispatchTableEntry kernel);

  RegistrationHandleRAII registerUnboxedAutogradKernel(void* kernel_func);

  const OperatorOptions& options() {
    return options_;
  }

private:
  void deregisterKernel_(TensorTypeId dispatch_key, std::list<DispatchTableEntry>::iterator kernel);
  void deregisterCatchallKernel_(std::list<DispatchTableEntry>::iterator kernel);
  void deregisterUnboxedAutogradKernel_(std::list<void*>::iterator kernel);

  FunctionSchema schema_;

  // The dispatchTable stores the current kernel for each dispatch key
  LeftRight<DispatchTable> dispatchTable_;

  // kernels_ is either:
  //   left:  a kernel map listing mapping from a dispatch key to a list of all
  //          kernels for that operator, or it is
  //   right: a list of all catch-all kernels registered for this operator.
  // An operator can only have either dispatched kernels or catch-all kernels,
  // not both.
  // In both cases, the list of kernels stores all registered kernels for the
  // corresponding dispatch key (or for catch-all).
  // If an operator library gets loaded that overwrites an already existing kernel,
  // both kernels will be in that list but only the newer one will be in
  // dispatchTable. If any of the kernels go away (say the library gets
  // unloaded), we remove the kernel from this list and update the
  // dispatchTable if necessary.
  // Kernels in the list are ordered by registration time descendingly,
  // newer registrations are before older registrations.
  // We do not combine dispatchTable and kernels into one hash map because
  // kernels is a larger data structure and accessed quite infrequently
  // while dispatchTable is accessed often and should be kept small to fit
  // into CPU caches.
  // Invariants (assuming kernels_.is_left()):
  //  - dispatchTable[dispatch_key] == kernels_.left()[dispatch_key].front()
  //  - dispatchTable[dispatch_key] does not exist if and only if
  //    kernels_.left()[dispatch_key] does not exist
  //  - If kernels_.left()[dispatch_key] exists, then it has elements.
  //    It is never an empty list.
  // Analogous invariants for kernels_.is_right().
  // The empty state (i.e. no kernels registered) is represented as an empty
  // map with kernels_.is_left().
  //
  // Why do we do that?
  // -----
  // We mostly do this to enable Jupyter notebooks where a cell registering
  // a kernel could be executed multiple times and the later execution
  // should overwrite the earlier one. Note that this still fails when the
  // function schema changed between the executions, but it works as long
  // as the function schema didn't change. A better solution would be to
  // unload the old extension library from the Jupyter cell when the cell is
  // re-ececuted and then only allow one kernel here, i.e. error if a kernel
  // is already registered, but that's a lot of effort to implement and
  // currently not high-pri.
  c10::either<
    ska::flat_hash_map<TensorTypeId, std::list<DispatchTableEntry>>, // dispatched kernels
    std::list<DispatchTableEntry> // catch-all kernels
  > kernels_;

  // unboxedAutogradKernels_ stores all autograd kernels registered for this op.
  // An autograd kernel has the same signature as the main op kernel and
  // internally re-dispatches to call the actual kernel.
  // Autograd kernels are unboxed currently. We are planning to move this
  // towards a system where ops register autograd wrappers (i.e. functions that
  // do some wrapping code and get a pointer to the actual kernel) instead of
  // autograd functions.
  // This is a list because, similar to kernels_, multiple libraries could
  // be loaded that register autograd kernels for the same op. The list is
  // ordered by registration time descendingly, i.e. newer registrations are
  // before older registrations and the list head is the autograd kernel
  // which is currently used.
  // See the comment for kernels_ above for an explanation for why we do this.
  std::list<void*> unboxedAutogradKernels_;
  std::atomic<void*> currentUnboxedAutogradKernel_;

  // Some metadata about the operator
  OperatorOptions options_;

  std::mutex kernelsMutex_; // protects kernels_
  std::mutex unboxedAutogradKernelsMutex_; // protects unboxedAutogradKernels_

  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given dispatch key.
  void updateDispatchTable_(TensorTypeId dispatch_key);
  void updateCatchallDispatchTable_();
  void updateCurrentUnboxedAutogradKernel_();
};

}
}
