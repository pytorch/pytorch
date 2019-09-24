#pragma once

#include <ATen/core/dispatch/DispatchTable.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <list>

namespace c10 {
namespace impl {
  class OperatorEntry;
}

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

  template<class Return, class... Args>
  Return callUnboxed(TensorTypeId dispatchKey, Args... args) const {
    // TODO Remove dispatchKey argument and instead infer dispatchKey from args...
    #if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && __GNUC__ < 5
      // GCC 4 has issues with parameter packs inside lambdas, let's instead
      // return the KernelFunction from the lambda. Note: This copies the
      // KernelFunction and is slow, but it's the only way to make it work for
      // GCC 4.
      KernelFunction func = dispatchTable_.read([&] (const DispatchTable& dispatchTable) -> KernelFunction {
          return dispatchTable.lookup(dispatchKey);
      });
      return func.callUnboxed<Return, Args...>(std::forward<Args>(args)...);
    #else
      // For all other compilers and newer GCC, let's do it right
      return dispatchTable_.read([&] (const DispatchTable& dispatchTable) -> Return {
          return dispatchTable.lookup(dispatchKey)
              .callUnboxed<Return, Args...>(std::forward<Args>(args)...);
      });
    #endif
  }

  template<class Return, class... Args>
  Return callUnboxedOnly(TensorTypeId dispatchKey, Args... args) const {
    // TODO Remove dispatchKey argument and instead infer dispatchKey from args...
    #if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && __GNUC__ < 5
      // GCC 4 has issues with parameter packs inside lambdas, let's instead
      // return the KernelFunction from the lambda. Note: This copies the
      // KernelFunction and is slow, but it's the only way to make it work for
      // GCC 4.
      KernelFunction func = dispatchTable_.read([&] (const DispatchTable& dispatchTable) -> KernelFunction {
          return dispatchTable.lookup(dispatchKey);
      });
      return func.callUnboxedOnly<Return, Args...>(std::forward<Args>(args)...);
    #else
      // For all other compilers and newer GCC, let's do it right
      return dispatchTable_.read([&] (const DispatchTable& dispatchTable) -> Return {
          return dispatchTable.lookup(dispatchKey)
              .callUnboxedOnly<Return, Args...>(std::forward<Args>(args)...);
      });
    #endif
  }

  void callBoxed(Stack* stack) const {
    return dispatchTable_.read([&] (const DispatchTable& dispatchTable) {
        dispatchTable.lookup(stack).callBoxed(stack);
    });
  }

  void prepareForDeregistration();

  RegistrationHandleRAII registerKernel(TensorTypeId dispatch_key, KernelFunction kernel);
  RegistrationHandleRAII registerCatchallKernel(KernelFunction kernel);

  const OperatorOptions& options() {
    return options_;
  }

private:
  void deregisterKernel_(TensorTypeId dispatch_key, std::list<KernelFunction>::iterator kernel);
  void deregisterCatchallKernel_(std::list<KernelFunction>::iterator kernel);

  FunctionSchema schema_;

  // The dispatchTable stores the current kernel for each dispatch key
  LeftRight<DispatchTable> dispatchTable_;

  // kernels_ stores all registered kernels for the corresponding dispatch key
  // and catchAllKernels_ stores the catch-all kernels.
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
  // Invariants:
  //  - dispatchTable[dispatch_key] == kernels_[dispatch_key].front()
  //  - dispatchTable[dispatch_key] does not exist if and only if
  //    kernels_[dispatch_key] does not exist
  //  - If kernels_[dispatch_key] exists, then it has elements.
  //    It is never an empty list.
  // Analogous invariants for catchAllKernels_.
  //
  // Why do we do that?
  // -----
  // We mostly do this to enable Jupyter notebooks where a cell registering
  // a kernel could be executed multiple times and the later execution
  // should overwrite the earlier one. Note that this still fails when the
  // function schema changed between the executions, but it works as long
  // as the function schema didn't change. A better solution would be to
  // unload the old extension library from the Jupyter cell when the cell is
  // re-executed and then only allow one kernel here, i.e. error if a kernel
  // is already registered, but that's a lot of effort to implement and
  // currently not high-pri.
  ska::flat_hash_map<TensorTypeId, std::list<KernelFunction>> kernels_;
  std::list<KernelFunction> catchAllKernels_;

  // Some metadata about the operator
  OperatorOptions options_;

  std::mutex kernelsMutex_; // protects kernels_

  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given dispatch key.
  void updateDispatchTable_(TensorTypeId dispatch_key);
  void updateCatchallDispatchTable_();
};

}
}
