#pragma once

#include <ATen/core/dispatch/DispatchTable.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <list>

namespace c10 {
namespace impl {

// This is a private class used inside the Dispatcher to represent an operator
// and it's dispatch table. This is not part of the public API.
class OperatorEntry final {
public:
  explicit OperatorEntry(FunctionSchema&& schema);

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
  RegistrationHandleRAII registerFallbackKernel(DispatchTableEntry kernel);

private:
  void deregisterKernel_(TensorTypeId dispatch_key, std::list<DispatchTableEntry>::iterator kernel);
  void deregisterFallbackKernel_();

  FunctionSchema schema_;

  // The dispatchTable stores the current kernel for each dispatch key
  LeftRight<DispatchTable> dispatchTable_;

  // The kernels map stores all registered kernels for a certain dispatch key.
  // If an operator library gets loaded that overwrites already existing kernels,
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
  //  - dispatchTable[dispatch_key] == kernels[dispatch_key].front()
  //  - dispatchTable[dispatch_key] does not exist if and only if
  //    kernels[dispatch_key] does not exist
  //  - If kernels[dispatch_key] exists, then it has elements.
  //    It is never an empty list.
  ska::flat_hash_map<TensorTypeId, std::list<DispatchTableEntry>> kernels_;
  std::mutex kernelsMutex_;

  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given dispatch key.
  void updateDispatchTable_(TensorTypeId dispatch_key);
};

}
}
