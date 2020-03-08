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
  explicit OperatorEntry(FunctionSchema&& schema);
  explicit OperatorEntry(OperatorName&& operator_name);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  const FunctionSchema& schema() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value());
    return *schema_;
  }

  // An OperatorEntry may be initialized with only an OperatorName.
  // If this is the case, we may post facto register a schema to it.
  //
  // Some rules:
  //  - The following programs are equivalent:
  //      OperatorEntry op(std::move(schema))
  //    and
  //      OperatorEntry op(schema.operator_name())
  //      op.registerSchema(std::move(schema))
  //  - The following programs are equivalent:
  //      OperatorEntry op(schema.operator_name())
  //    and
  //      OperatorEntry op(std::move(schema))
  //      op.deregisterSchema()
  //
  // NB: registerSchema/deregisterSchema are not idempotent; if you
  // attempt to register a schema when one is already present or vice
  // versa that is an error.  (Refcounting for the registrations is
  // handled in the OperatorHandle in Dispatcher)
  void registerSchema(FunctionSchema&&);
  void deregisterSchema();

  const OperatorName& operator_name() const {
    return name_;
  }

  const DispatchTable& dispatch_table() const {
    return dispatchTable_;
  }

  void prepareForDeregistration();

  // Postcondition: caller is responsible for disposing of the kernel
  std::list<KernelFunction>::iterator registerKernel(c10::optional<DispatchKey> dispatch_key, KernelFunction kernel);
  void deregisterKernel_(c10::optional<DispatchKey> dispatch_key, std::list<KernelFunction>::iterator kernel);

private:

  OperatorName name_;
  c10::optional<FunctionSchema> schema_;

  // The dispatchTable stores the current kernel for each dispatch key
  DispatchTable dispatchTable_;

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
  ska::flat_hash_map<c10::optional<DispatchKey>, std::list<KernelFunction>> kernels_;

  std::mutex kernelsMutex_; // protects kernels_

  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given dispatch key.
  void updateDispatchTable_(c10::optional<DispatchKey> dispatch_key);
};

}
}
