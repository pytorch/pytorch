#pragma once

#include <ATen/core/function_schema.h>
#include <c10/util/Metaprogramming.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/either.h>
#include <c10/util/Optional.h>
#include <c10/core/DispatchKey.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/DispatchKeyExtractor.h>

#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/dispatch/CppSignature.h>
#include <ATen/core/dispatch/RegistrationHandleRAII.h>

#include <list>
#include <array>

#ifdef C10_MOBILE
#define C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
#endif

namespace c10 {

class Dispatcher;

namespace impl {

// This data structure represents a kernel that was registered to us from a
// user.  Unlike KernelFunction, AnnotatedKernel contains some extra metadata
// about the kernel that isn't necessary for actual dispatching (this is why
// we don't put AnnotatedKernel in the actual DispatchTable), but is useful for
// giving good error messages.
struct AnnotatedKernel final {
  AnnotatedKernel(KernelFunction k, std::unique_ptr<FunctionSchema> s, std::string d)
    : kernel(std::move(k))
    , inferred_function_schema(std::move(s))
    , debug(std::move(d))
    {}
  AnnotatedKernel() {}
  KernelFunction kernel;
  std::unique_ptr<FunctionSchema> inferred_function_schema;
  // A little debug string to help us identify the kernel in question.
  // Most importantly it records the TORCH_LIBRARY block that did the
  // registration.
  std::string debug;
};

// This data structure represents operator schema, with metadata specifying
// where the registration of this schema occurred
struct AnnotatedSchema final {
  AnnotatedSchema(FunctionSchema s, std::string d)
    : schema(std::move(s))
    , debug(std::move(d))
    {}
  FunctionSchema schema;
  std::string debug;
};

// Internal data structure that records information about a specific operator.
// It's not part of the public API; typically, users will interact with
// OperatorHandle instead.
//
// Concurrent writes to OperatorEntry are protected by the GLOBAL Dispatcher
// lock (this is important because some methods in OperatorEntry access
// dispatcher state)
class TORCH_API OperatorEntry final {
public:
  explicit OperatorEntry(OperatorName&& operator_name);

  OperatorEntry(const OperatorEntry&) = delete;
  OperatorEntry(OperatorEntry&&) noexcept = delete;
  OperatorEntry& operator=(const OperatorEntry&) = delete;
  OperatorEntry& operator=(OperatorEntry&&) noexcept = delete;

  const FunctionSchema& schema() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value(), "Tried to access the schema for ", name_, " which doesn't have a schema registered yet");
    return schema_->schema;
  }
  const std::string& debug() const {
    TORCH_INTERNAL_ASSERT(schema_.has_value());
    return schema_->debug;
  }
  bool hasSchema() const {
    return schema_.has_value();
  }

  bool isObserved() const {
    return is_observed_;
  }

  // We may allocate an OperatorEntry for an operator even when we don't
  // have a schema.  When we receive the schema registration, we post
  // facto register a schema.
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

#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
  using AnnotatedKernelContainer = std::array<AnnotatedKernel, 1>;
#else
  using AnnotatedKernelContainer = std::list<AnnotatedKernel>;
#endif
  using AnnotatedKernelContainerIterator = AnnotatedKernelContainer::iterator;

  // Why are kernels and fallback asymmetric?  It has to do with ownership.
  // Kernels and the computed dispatch tables for them are canonically
  // owned by OperatorEntry, but backend fallbacks are specified once
  // and apply for all operators, so they should be owned by Dispatcher.
  // However, the registration of a backend fallback affects the
  // state of the computed dispatch table, so when a backend fallback
  // is updated, we need to update the operator tables too.  Thus,
  // registerKernel is the mechanism by which we give kernels to
  // operator entry to own (and update dispatch table), but we only
  // need a non-owning mechanism to update fallback.

  // Precondition: Dispatcher::mutex_ is held
  // Postcondition: caller is responsible for disposing of the kernel
  AnnotatedKernelContainerIterator registerKernel(
    const Dispatcher& dispatcher,
    c10::optional<DispatchKey> dispatch_key,
    KernelFunction kernel,
    c10::optional<CppSignature> cpp_signature,
    std::unique_ptr<FunctionSchema> inferred_function_schema,
    std::string debug
  );

  // Precondition: Dispatcher::mutex_ is held
  void deregisterKernel_(
    const Dispatcher& dispatcher,
    c10::optional<DispatchKey> dispatch_key,
    AnnotatedKernelContainerIterator kernel
  );

  // Precondition: Dispatcher::mutex_ is held
  void updateFallback(
    const Dispatcher& dispatcher,
    DispatchKey dispatch_key
  );

  // Precondition: Dispatcher::mutex_ is held
  void updateSchemaAliasAnalysis(AliasAnalysisKind a) {
    TORCH_INTERNAL_ASSERT(schema_.has_value());
    schema_->schema.setAliasAnalysis(a);
  }

  std::string dumpComputedTable() const;
  std::string dumpState() const;
  void checkInvariants() const;

  const DispatchKeyExtractor& dispatchKeyExtractor() const { return dispatchKeyExtractor_; }

  // Asserts that the given FuncType is correct for calling this operator in an unboxed way.
  template<class FuncType>
  inline void assertSignatureIsCorrect() {
    assertSignatureIsCorrect(CppSignature::make<FuncType>());
  }

  void assertSignatureIsCorrect(const CppSignature call_signature) {
    if (C10_UNLIKELY(cpp_signature_.has_value() && (call_signature != cpp_signature_->signature))) {
      reportSignatureError(call_signature);
    }
  }

  [[noreturn]] void reportError(DispatchKey dispatchKey) const;

  const KernelFunction& lookup(DispatchKey k) const {
    const auto idx = getDispatchTableIndexForDispatchKey(k);
    if (C10_UNLIKELY(idx == -1)) {
      reportError(k);
    }
    const auto& kernel = dispatchTable_[idx];
    // A valid kernel *always* has a boxed kernel and *may* have an
    // unboxed kernel. However, we typically do unboxed calls in at::
    // APIs, where the kernel 1) will very likely be valid and 2)
    // should have an unboxed kernel. Checking the unboxed kernel
    // first will allow us to avoid touching the boxed kernel at all
    // in the common case.
    if (C10_UNLIKELY(!kernel.isValidUnboxed())) {
      if (!kernel.isValid()) {
        reportError(k);
      }
    }
    return kernel;
  }

  std::string listAllDispatchKeys() const;

  // Returns true if kernel_ has entry for any key in ks.
  //
  // Invariant: There are no alias keys in the passed-in dispatch key set.
  // Note [No Alias Keys in DispatchKeySet]
  // Alias keys should be checked using `hasKernelForDispatchKey`
  // Alias keys shouldn't go inside of a DispatchKeySet, since they can technically
  // have a value > 63 (causing overflow).
  bool hasKernelForAnyDispatchKey(DispatchKeySet ks) const;
  // Returns true if kernel_ has entry for a particular key.
  bool hasKernelForDispatchKey(DispatchKey k) const;

private:

  OperatorName name_;
  c10::optional<AnnotatedSchema> schema_;

  std::array<KernelFunction, c10::getDispatchTableIndexForDispatchKey(DispatchKey::NumDispatchKeys)> dispatchTable_;
  DispatchKeyExtractor dispatchKeyExtractor_;

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
  ska::flat_hash_map<DispatchKey,
#ifdef C10_DISPATCHER_ONE_KERNEL_PER_DISPATCH_KEY
                     // On mobile, we needn't worry about Jupyter notebooks.
                     std::array<AnnotatedKernel, 1>
#else
                     std::list<AnnotatedKernel>
#endif
                     > kernels_;

  const AnnotatedKernel& missingKernel() const;
  const AnnotatedKernel& ambiguousAutogradOtherKernel() const;

  // cpp_signature_ stores function signature if any of
  // the kernels was created in a way that allowed us to know the function
  // signature (i.e. by supplying an unboxed C++ kernel function).
  // If this is set, it will be used to check that future kernel
  // registrations match and it will be used in unboxed function calls
  // to verify their arguments against the known function signature.
  struct CppSignatureWithDebug {
    CppSignature signature;
    std::string debug;
    c10::optional<DispatchKey> dispatch_key;
  };
  c10::optional<CppSignatureWithDebug> cpp_signature_;

  // Whether this operator needs to be observed with RecordFunction
  const bool is_observed_;

  [[noreturn]] void reportSignatureError(CppSignature call_signature) const;
  const KernelFunction& computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const;
  std::pair<const AnnotatedKernel&, const char*> computeDispatchTableEntryWithDebug(
    const c10::Dispatcher& dispatcher, DispatchKey dispatch_key
  ) const;
  // This function re-establishes the invariant that dispatchTable
  // contains the front element from the kernels list for a given runtime dispatch key.
  void updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key);
  // Like above, but also handles alias dispatch keys.
  void updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key);
  // Like above, but for ALL entries in the dispatch table.
  void updateDispatchTableFull_(const c10::Dispatcher& dispatcher);
  // Retrieves a pointer to AnnotatedKernel at kernels_.at(dispatch_key).front().
  const AnnotatedKernel* getKernelForDispatchKey(DispatchKey dispatch_key) const;
};

} // namespace impl
} // namespace c10
