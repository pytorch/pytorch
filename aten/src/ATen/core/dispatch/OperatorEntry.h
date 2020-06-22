#pragma once

#include <ATen/core/dispatch/DispatchTable.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>
#include <list>

namespace c10 {
namespace impl {
  class OperatorEntry;
}

namespace impl {

// This is a private class used inside the Dispatcher to represent an operator
// and its dispatch table. This is not part of the public API.
class CAFFE2_API OperatorEntry final {
public:
  struct KernelEntry final {
    KernelEntry(KernelFunction k, std::unique_ptr<FunctionSchema> s, std::string d)
      : kernel(std::move(k))
      , inferred_function_schema(std::move(s))
      , debug(std::move(d))
      {}
    KernelFunction kernel;
    std::unique_ptr<FunctionSchema> inferred_function_schema;
    // A little debug string to help us identify the kernel in question.
    // Mostly used in testing but it might be possible to augment
    // regular registrations with some more info here too
    std::string debug;
  };

  explicit OperatorEntry(OperatorName&& operator_name);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  const FunctionSchema& schema() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value(), "Tried to access the schema for ", name_, " which doesn't have a schema registered yet");
    return *schema_;
  }
  const std::string& debug() const {
    TORCH_INTERNAL_ASSERT(debug_.has_value());
    return *debug_;
  }
  bool hasSchema() const {
    return schema_.has_value();
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
  void registerSchema(FunctionSchema&&, std::string&& debug);
  void deregisterSchema();

  const OperatorName& operator_name() const {
    return name_;
  }

  const DispatchTable& dispatch_table() const {
    return dispatchTable_;
  }

  void prepareForDeregistration();

  // Postcondition: caller is responsible for disposing of the kernel
  std::list<KernelEntry>::iterator registerKernel(c10::optional<DispatchKey> dispatch_key, KernelFunction kernel, c10::optional<CppSignature> cpp_signature, std::unique_ptr<FunctionSchema> inferred_function_schema, std::string debug);
  void deregisterKernel_(c10::optional<DispatchKey> dispatch_key, std::list<KernelEntry>::iterator kernel);

  void updateSchemaAliasAnalysis(AliasAnalysisKind a) {
    TORCH_INTERNAL_ASSERT(schema_.has_value());
    schema_->setAliasAnalysis(a);
  }

  std::string dumpState() const;
  void checkInvariants() const;

  // This function is a temporary hack that allows generated_unboxing_wrappers.cpp to register its codegen'ed
  // unboxing wrapper for aten operators. We still need those for some operators because not all work
  // with the templated unboxing logic yet.
  // TODO Delete setManuallyBoxedKernel_ once all operators work with the templated boxing logic
  void setManuallyBoxedKernel_(KernelFunction::InternalBoxedKernelFunction* func) {
    dispatchTable_.setManuallyBoxedKernel_(func);
  }

  // Asserts that the given FuncType is correct for calling this operator in an unboxed way.
  template<class FuncType>
  void assertSignatureIsCorrect() {
    TORCH_INTERNAL_ASSERT(!cpp_signature_.has_value() || (CppSignature::make<FuncType>() == *cpp_signature_),
        "Tried to access operator ", name_, " with a wrong signature. Accessed with ",
        CppSignature::make<FuncType>().name(),
        " but the operator was registered with ",
        cpp_signature_->name(),
        " (",
        debug_.value(),
        ") This likely happened in a call to OperatorHandle::typed<Return (Args...)>(). Please make sure that the function signature matches the signature in the operator registration call."
    );
  }

private:

  OperatorName name_;
  c10::optional<FunctionSchema> schema_;
  c10::optional<std::string> debug_;
  // INVARIANT: schema_.has_value() == debug_.has_value()

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
  ska::flat_hash_map<c10::optional<DispatchKey>, std::list<KernelEntry>> kernels_;

  std::mutex kernelsMutex_; // protects kernels_

  // signature_hash_ is set to the hash of the function signature if any of
  // the kernels was created in a way that allowed us to know the function
  // signature (i.e. by supplying an unboxed C++ kernel function).
  // If this is set, it will be used in unboxed function calls
  // to verify their arguments against the known function signature.
  c10::optional<CppSignature> cpp_signature_;

  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given dispatch key.
  void updateDispatchTable_(c10::optional<DispatchKey> dispatch_key);
};

}
}
