#pragma once

#include <ATen/core/dispatch/DispatchTable.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <list>

namespace c10 {
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

  DispatchTableEntry lookupKernel(const Stack* stack) const {
    return dispatchTable_.read([&] (const DispatchTable& dispatchTable) {
      return dispatchTable.lookup(stack);
    });
  }

  void prepareForDeregistration();

  RegistrationHandleRAII registerKernel(TensorTypeId dispatch_key, DispatchTableEntry kernel);
  RegistrationHandleRAII registerCatchallKernel(DispatchTableEntry kernel);

  const OperatorOptions& options() {
    return options_;
  }

private:
  void deregisterKernel_(TensorTypeId dispatch_key, std::list<DispatchTableEntry>::iterator kernel);
  void deregisterCatchallKernel_(std::list<DispatchTableEntry>::iterator kernel);

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
  c10::either<
    ska::flat_hash_map<TensorTypeId, std::list<DispatchTableEntry>>, // dispatched kernels
    std::list<DispatchTableEntry> // catch-all kernels
  > kernels_;
  std::mutex kernelsMutex_; // protects kernels_

  // Some metadata about the operator
  OperatorOptions options_;

  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given dispatch key.
  void updateDispatchTable_(TensorTypeId dispatch_key);
  void updateCatchallDispatchTable_();
};

}
}
