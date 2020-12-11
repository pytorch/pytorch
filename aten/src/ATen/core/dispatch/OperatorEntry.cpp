#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/dispatch/ObservedOperators.h>

namespace c10 {
namespace impl {

namespace {
  std::string toString(c10::optional<DispatchKey> k) {
    if (k.has_value()) {
      return toString(*k);
    } else {
      return "(catch all)";
    }
  }
}

OperatorEntry::OperatorEntry(OperatorName&& operator_name)
: name_(std::move(operator_name))
, schema_()
, dispatchTable_()
, dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized())
, kernels_()
, cpp_signature_()
, is_observed_(ObservedOperators::isObserved(name_))
{
  // Pick up any backend fallbacks that were registered prior to this
  // OperatorEntry being created
  updateDispatchTableFull_(c10::Dispatcher::singleton());
}

namespace {
  void checkSchema(const OperatorName& name, const FunctionSchema& from_def, const std::string& from_def_debug, const FunctionSchema& inferred, const std::string& inferred_debug) {
    c10::optional<std::string> schema_difference = findSchemaDifferences(from_def, inferred);
    if (schema_difference.has_value()) {
      TORCH_CHECK(false,
        "Inferred operator schema for a C++ kernel function doesn't match the expected function schema.\n"
        "  operator: ", toString(name), "\n",
        "  expected schema: ", toString(from_def), "\n",
        "    ", from_def_debug, "\n",
        "  inferred schema: ", toString(inferred), "\n",
        "    ", inferred_debug, "\n",
        "  reason: ", *schema_difference);
    }
  }
} // anonymous namespace

const AnnotatedKernel OperatorEntry::ambiguousAutogradOtherKernel_ = AnnotatedKernel(
    c10::KernelFunction::makeAmbiguousAutogradOther(), nullptr, "ambiguous_autogradother");

void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug) {
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  for (auto i = kernels_.begin(); i != kernels_.end(); ++i) {
    for (auto j = i->second.begin(); j != i->second.end(); ++j) {
      if (j->inferred_function_schema != nullptr) {
        checkSchema(name_, schema, debug, *j->inferred_function_schema, j->debug);
      }
    }
  }
  // NB: don't register schema until after we've checked everything!
  dispatchKeyExtractor_.registerSchema(schema);
  schema_ = AnnotatedSchema(std::move(schema), std::move(debug));
}

void OperatorEntry::deregisterSchema() {
  TORCH_INTERNAL_ASSERT(schema_.has_value());
  schema_ = c10::nullopt;
  dispatchKeyExtractor_.deregisterSchema();
}

std::list<AnnotatedKernel>::iterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  // NB: cpp_signature doesn't get cleared even after the kernel that populated
  // it is deleted.  This means you could poison the value of cpp_signature_
  // with a bad signature value, and then it would permanently stay there until
  // you deregister the schema.  This can't really be fixed, because we
  // only do a typed() test once in the lifetime of a TypedOperatorHandle,
  // which means if you could validly change the type of a cpp_signature, then
  // that would also invalidate the old TypedOperatorHandles.
  if (cpp_signature.has_value()) {
    if (cpp_signature_.has_value()) {
      TORCH_CHECK(*cpp_signature == cpp_signature_->signature,
        "\nMismatch in kernel C++ signatures\n",
        "  operator: ", (this->schema_.has_value() ? toString(this->schema_->schema) : toString(name_)), "\n",
        "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
        "  kernel 1: ", cpp_signature_->signature.name(), "\n",
        "    dispatch key: ", toString(cpp_signature_->dispatch_key), "\n",
        "    ", cpp_signature_->debug, "\n",
        "  kernel 2: ", cpp_signature->name(), "\n",
        "    dispatch key: ", toString(dispatch_key), "\n",
        "    ", debug, "\n"
      );
    } else {
      cpp_signature_ = CppSignatureWithDebug { *cpp_signature, debug, dispatch_key };
    }
  }

  if (schema_ && inferred_function_schema) {
    checkSchema(name_, schema_->schema, schema_->debug, *inferred_function_schema, debug);
  }

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  // Redirect catchAll registrations to Math.
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::Math];

  if (k.size() > 0) {
    TORCH_WARN("Overriding a previously registered kernel for the same operator and the same dispatch key\n",
               "  operator: ", (schema_.has_value() ? toString(schema_->schema) : toString(name_)), "\n",
               "    ", (this->schema_.has_value() ? this->schema_->debug : "no debug info"), "\n",
               "  dispatch key: ", toString(dispatch_key), "\n",
               "  previous kernel: ", (cpp_signature_.has_value() ? cpp_signature_->debug : "no debug info"), "\n",
               "       new kernel: ", debug
    );
  }

  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
  std::list<AnnotatedKernel>::iterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;
}

void OperatorEntry::deregisterKernel_(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  std::list<AnnotatedKernel>::iterator kernel
) {
  // Redirect catchAll deregistrations to Math.
  DispatchKey dk = dispatch_key.has_value() ? *dispatch_key : DispatchKey::Math;
  auto found = kernels_.find(dk);
  TORCH_INTERNAL_ASSERT(found != kernels_.end(), "Tried to deregister a kernel for dispatch key ", toString(dispatch_key), " but there are no kernels registered for this dispatch key. The operator is ", toString(name_));
  auto& k = found->second;
  k.erase(kernel);
  if (k.empty()) {
    // the invariant says we don't want empty lists but instead remove the list from the map
    kernels_.erase(found);
  }
  updateDispatchTable_(dispatcher, dk);
}

void OperatorEntry::updateFallback(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  updateDispatchTable_(dispatcher, dispatch_key);
}

const KernelFunction& OperatorEntry::computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  return computeDispatchTableEntryWithDebug(dispatcher, dispatch_key).first.kernel;
}

bool OperatorEntry::hasKernelForAnyDispatchKey(DispatchKeySet ks) const {
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
  for (auto& kv : kernels_) {
    if (ks.has(kv.first)) return true;
  }
  return false;
}

c10::optional<const AnnotatedKernel*> OperatorEntry::getKernelForDispatchKey(DispatchKey dispatch_key) const{
  auto kern_it = kernels_.find(dispatch_key);
  if (kern_it != kernels_.end()) {
    TORCH_INTERNAL_ASSERT(!kernels_.at(dispatch_key).empty());
    TORCH_INTERNAL_ASSERT(kernels_.at(dispatch_key).front().kernel.isValid());
    return c10::make_optional(&kernels_.at(dispatch_key).front());
  }
  return c10::nullopt;
}

std::pair<const AnnotatedKernel&, const char*> OperatorEntry::computeDispatchTableEntryWithDebug(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  // [Note] DispatchTable computation
  // dispatchTable contains entries for runtime dispatch keys.
  // For any dispatch key, it'll pick a kernel using the following order:
  //  (1) Use kernel if it's directly registered to this key
  //  (2) Handle runtime keys that have kernels available from alias keys
  //    (2.1) Use kernel from DispatchKey::DefaultBackend if available.
  //          This is used to register a kernel that works for all backend in inference. But it requires
  //          separate registration for Autograd keys to support training.
  //    (2.2) Use kernel from DispatchKey::Math if available.
  //          For autograd keys, we only use kernel from Math when there's no direct registration
  //          to its corresponding backend key or DefaultBackend. See Note [DefaultBackend and Math].
  //          For AutogradOther, we eagerly return ambiguousAutogradOtherKernel_ if there's registration to any of
  //          its backends and ask backend extender to request a decicated Autograd key for the backend.
  //          See Note [Ambiguity in AutogradOther kernel] for more details.
  //          A DefaultBackend kernel prevents Math kernel being used for Autograd keys, but it doesn't
  //          cause confusion for AutogradOther. It's pretty straightforward to use Autograd (if available)
  //          in this case.
  //    (2.3) Use kernel from DispatchKey::Autograd if available
  //    The implementation of (2.2) relies on the invariant that for a given backend,
  //    `computeDispatchTableEntryWithDebug()` will be called for that backend's autograd key after the
  //    backend key. See Note [Refresh Runtime Autograd entries in dispatchTable_]
  //  (3) Use fallthrough kernel that are registered as fallback.
  // Alias Key Precedence:
  //   DefaultBackend > Math > Autograd
  // Note [DefaultBackend and Math]
  //   When there're registrations to both DefaultBackend & Math & Autograd, from (2.2) we know DefaultBackend
  //   and Autograd kernels will be picked up and Math is overriden.
  //   This is fine and in practice DefaultBackend and Math shouldn't co-exist for an op.
  // TODO: Update alias key precedence after we add new alias keys AutogradDispatchCPUOrCUDA .

  // 1. Operator registration
  if (auto direct_registration = getKernelForDispatchKey(dispatch_key)) {
    return {*direct_registration.value(), "kernel"};
  }

  // 2.1 Use DefaultBackend kernel if available.
  //     See Note [Undefined in dispatchTable_] for the special handling for Undefined.
  if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::DefaultBackend)) {
    if (auto default_backend_registration = getKernelForDispatchKey(DispatchKey::DefaultBackend)) {
      return {*default_backend_registration.value(), "default backend kernel"};
    }
  }

  // Note when there's direct registration to DefaultBackend, this code path will only be hit by
  // non backend keys (e.g AutogradXXX, Batched etc) due to (2.1).
  bool has_backend_kernel =
    hasKernelForAnyDispatchKey(getBackendKeySetFromAutograd(dispatch_key).add(DispatchKey::DefaultBackend));

  // 2.2. Use Math kernel if available. For autograd keys, we only use kernel from Math
  //      when there's no direct registration to its corresponding backend key or DefaultBackend.
  //      For AutogradOther, we return ambiguousAutogradOtherKernel_ if there's registration
  //      to any of its backends.
  //      See Note [Undefined in dispatchTable_] for the special handling for Undefined.
  if (dispatch_key == DispatchKey::Undefined || isIncludedInAlias(dispatch_key, DispatchKey::Math)) {
    if (auto math_registration = getKernelForDispatchKey(DispatchKey::Math)) {
      if (dispatch_key == DispatchKey::AutogradOther
          && hasKernelForAnyDispatchKey(c10::autogradother_backends)) {
        return {ambiguousAutogradOtherKernel_, "ambiguous autogradother"};
      } else if (!has_backend_kernel) {
        return {*math_registration.value(), "math kernel"};
      }
    }
  }

  // 2.3. For autograd backend keys, use kernel from DispatchKey::Autograd if available
  if (isIncludedInAlias(dispatch_key, DispatchKey::Autograd)) {
    if (auto autograd_registration = getKernelForDispatchKey(DispatchKey::Autograd)) {
      return {*autograd_registration.value(), "autograd kernel"};
    }
  }

  // 3. Backend fallback
  auto dispatch_ix = static_cast<uint8_t>(dispatch_key);
  if (dispatcher.backendFallbackKernels_[dispatch_ix].kernel.isValid()) {
    return {dispatcher.backendFallbackKernels_[dispatch_ix], "backend fallback"};
  }

  // 4. Default to error
  return {missingKernel_, "missing"};
}

// synchronizes the dispatch table entry for a given dispatch key
// with the current state of kernel registrations in the dispatcher.
// note that this is not a complete update, due to relationships between
// dispatch keys (e.g. runtime keys and their associated autograd keys,
// or alias keys and their associated keysets).
// This function should be considered a private helper for updateDispatchTable_()
void OperatorEntry::updateDispatchTableEntry_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  auto dispatch_ix = static_cast<uint8_t>(dispatch_key);
  dispatchTable_[dispatch_ix] = computeDispatchTableEntry(dispatcher, dispatch_key);
  dispatchKeyExtractor_.setOperatorHasFallthroughForKey(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());
}

// synchronizes the dispatch table entries for a given dispatch key *and its
// associated keys* with the current state of kernel registrations in the
// dispatcher.
// After a kernel has been registered to a dispatch key, a call to this
// function will synchronize the dispatcher state. See e.g. registerKernel()
void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  // Handle Undefined separately since it isn't a runtime key but we have an entry in dispatchTable_.
  // See Note [Undefined in dispatchTable_]
  if (dispatch_key == DispatchKey::Undefined) {
    updateDispatchTableEntry_(dispatcher, dispatch_key);
    return;
  }
  for (auto k : c10::getRuntimeDispatchKeySet(dispatch_key)) {
    updateDispatchTableEntry_(dispatcher, k);
  }
  // Registration to DefaultBackend and Math should be populated to Undefined.
  // We cannot do this above since Undefined cannot be represented in DispatchKeySet.
  if (dispatch_key == DispatchKey::Math || dispatch_key == DispatchKey::DefaultBackend) {
    updateDispatchTableEntry_(dispatcher, DispatchKey::Undefined);
  }
  // Note [Refresh Runtime Autograd entries in dispatchTable_]
  // Registering to backend key might affect computed entry at its Autograd backend key due to (2.1) & (2.3).
  if (c10::isBackendDispatchKey(dispatch_key)) {
    DispatchKey autograd_key = getAutogradKeyFromBackend(dispatch_key);
    updateDispatchTableEntry_(dispatcher, autograd_key);
  }
}

// does a complete update of the dispatch table, synchronizing all
// runtime dispatch keys with the current state of kernel registrations
// in the dispatcher.
// Note that we use updateDispatchTable_() to perform our per-key updating,
// even though that function is equipped to handle out-of-order updates and
// alias key updates, neither of which we send it. This is deliberate - the
// current design is more tractable with all updates funneled through a single
// per-key update mechanism, than with multiple variations that assume different
// invariants.
//
void OperatorEntry::updateDispatchTableFull_(const c10::Dispatcher& dispatcher) {
  // Note [Undefined in dispatchTable_]
  // DispatchKey Undefined is used in runtime:
  // (1) it gives people place to specify functionality that should run when there are no dispatch keys,
  //     e.g., an op without Tensor inputs or empty TensorList arguments
  // (2) it would let us remove the explicit error checking code in the dispatch hotpath, and so when
  //     no dispatch keys are available we just slide into the undefined handler which would then raise
  //     the error message.
  // In the old world of catchAll, the only way to "register" a kernel to Undefined is by registering it to
  // catchAll. After catchAllKernel_ is removed, Undefined now can get a kernel from either DefaultBackend
  // or Math alias key so that we don't break the support. Ideally isIncludedInAlias(Undefined, Math)
  // should return true, it returns false because Undefined cannot be represented in a DispatchKeySet.
  for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
    updateDispatchTable_(dispatcher, static_cast<DispatchKey>(iter));
  }
}

void OperatorEntry::checkInvariants() const {
  if (schema_) {
    TORCH_INTERNAL_ASSERT(schema_->schema.operator_name() == name_, dumpState());
    dispatchKeyExtractor().checkInvariants(schema_->schema);
  }
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end(), dumpState());
  for (const auto& kv : kernels_) {
    TORCH_INTERNAL_ASSERT(kv.second.size() > 0, dumpState());
  }
  for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
    auto expected_k = computeDispatchTableEntry(c10::Dispatcher::singleton(), static_cast<DispatchKey>(iter));
    TORCH_INTERNAL_ASSERT(expected_k._equalsBoxedAndUnboxed(dispatchTable_[iter]),
      "Canonical state\n~~~~~~~~~~~\n", dumpState(), "\n\n"
      "Computed table:\n~~~~~~~~~~~\n", dumpComputedTable());
  }
}

std::string OperatorEntry::listAllDispatchKeys() const {
  std::ostringstream str;
  str << "[";

  bool has_kernels = false;
  for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
    if (!dispatchTable_[iter].isValid()) {
      continue;
    }
    if (has_kernels) {
      str << ", ";
    }
    str << static_cast<DispatchKey>(iter);
    has_kernels = true;
  }
  str << "]";
  return str.str();
}

void OperatorEntry::reportError(DispatchKey dispatchKey) const {
  // If there is an invariant problem, report it now.
  checkInvariants();

  if (dispatchKey == DispatchKey::Undefined) {
    TORCH_CHECK(false,
          "There were no tensor arguments to this function (e.g., you passed an "
          "empty list of Tensors), but no fallback function is registered for schema ", name_,
          ".  This usually means that this function requires a non-empty list of Tensors.  "
          "Available functions are ", listAllDispatchKeys(), ".\n\n", dumpComputedTable())
  }

  TORCH_CHECK(false, "Could not run '", name_, "' with arguments",
          " from the '", toString(dispatchKey), "' backend. This could be because "
          "the operator doesn't exist for this backend, or was omitted during ",
          "the selective/custom build process (if using custom build). If you are a ",
          "Facebook employee using PyTorch on mobile, please visit ",
          "https://fburl.com/ptmfixes for possible resolutions. '",
          name_, "' is only available for these backends: ",
          listAllDispatchKeys(), ".\n\n", dumpComputedTable());
}

// INSPECTING DISPATCHER STATE
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The dumper functions purposely do not check invariants, as you might be using
// them to debug situations where the invariants are violated.

// Inspect what the computed dispatch table would be (e.g., what
// updateDispatchTableFull_ would update the dispatch table to be)
std::string OperatorEntry::dumpComputedTable() const {
  std::ostringstream oss;
  for (uint8_t i = 0; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys); i++) {
    auto k = static_cast<DispatchKey>(i);
    auto kernel_prov = computeDispatchTableEntryWithDebug(c10::Dispatcher::singleton(), k);
    if (kernel_prov.first.kernel.isValid()) {
      oss << toString(k) << ": "
          << (kernel_prov.first.kernel.isFallthrough() ? "fallthrough " : "")
          << kernel_prov.first.debug << " [" << kernel_prov.second << "]\n";
    }
  }
  return oss.str();
}

// Inspect the "canonical" information in OperatorEntry.  This only prints out
// *non-derived* information including kernels registered to alias dispatch keys;
// i.e., what the source of truth says about the operator.  This dumping function
// is appropriate for expect tests.
// This WON'T report backend fallbacks.
std::string OperatorEntry::dumpState() const {
  std::ostringstream oss;
  oss << "name: " << name_ << "\n";
  if (schema_) {
    oss << "schema: " << schema_->schema << "\n";
    oss << "debug: " << schema_->debug << "\n";
    oss << "alias analysis kind: " << toString(schema_->schema.aliasAnalysis())
        << (schema_->schema.isDefaultAliasAnalysisKind() ? " (default)" : "") << "\n";
  } else {
    oss << "schema: (none)\n";
  }

  auto print_kernel = [&](const char* k_desc, const std::list<AnnotatedKernel>& jts, bool is_alias_key=false) {
    int64_t i = 0;
    for (const auto& jt : jts) {
      oss << k_desc
          << (is_alias_key ? "[alias]" :  "")
          << (i > 0 ? " (inactive)" : "")
          << ": "
          << jt.debug << " :: "
          << (jt.inferred_function_schema ? toString(*jt.inferred_function_schema) : "(none)")
          << " [ " << jt.kernel.dumpState() << "]\n";
      i++;
    }
  };

  // Iterate over DispatchKey, not the flat hash map, so we have a stable order
  for (uint8_t i = 0; i <= static_cast<uint8_t>(DispatchKey::EndOfAliasKeys); i++) {
    auto k = static_cast<DispatchKey>(i);
    auto it = kernels_.find(k);
    if (it != kernels_.end()) {
      print_kernel(toString(k), it->second, c10::isAliasDispatchKey(k));
    }
  }
  return oss.str();
}

}
}
