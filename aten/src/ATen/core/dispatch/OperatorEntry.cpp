#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/op_registration/infer_schema.h>
#include <ATen/core/dispatch/Dispatcher.h>

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

  std::string listAllDispatchKeys(const ska::flat_hash_map<c10::optional<DispatchKey>, std::list<AnnotatedKernel>>& kernels) {
    if (kernels.size() == 0) {
      return "";
    }
    std::ostringstream str;
    str << toString(kernels.begin()->first);
    for (auto iter = ++kernels.begin(); iter != kernels.end(); ++iter) {
      str << ", " << toString(iter->first);
    }
    return str.str();
  }
}

OperatorEntry::OperatorEntry(OperatorName&& operator_name)
: name_(std::move(operator_name))
, schema_()
, dispatchTable_()
, dispatchKeyExtractor_(DispatchKeyExtractor::makeUninitialized())
, manuallyBoxedKernel_()
, kernels_()
, catchAllKernel_()
, cpp_signature_()
{}

namespace {
  void checkSchema(const OperatorName& name, const FunctionSchema& from_def, const std::string& from_def_debug, const FunctionSchema& inferred, const std::string& inferred_debug) {
    c10::optional<std::string> schema_difference = findSchemaDifferences(from_def, inferred);
    if (schema_difference.has_value()) {
      TORCH_CHECK(false,
        "In registration for ", toString(name), ": expected schema of operator to be \"", toString(from_def), "\" (", from_def_debug, "), ",
        "but got inferred schema \"", toString(inferred), "\" (", inferred_debug, "). ",
        *schema_difference);
    }
  }
} // anonymous namespace

void OperatorEntry::registerSchema(FunctionSchema&& schema, std::string&& debug) {
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  for (auto i = kernels_.begin(); i != kernels_.end(); ++i) {
    for (auto j = i->second.begin(); j != i->second.end(); ++j) {
      if (j->inferred_function_schema != nullptr) {
        checkSchema(name_, schema, debug, *j->inferred_function_schema, j->debug);
      }
    }
  }
  for (auto j = catchAllKernel_.begin(); j != catchAllKernel_.end(); ++j) {
    if (j->inferred_function_schema != nullptr) {
      checkSchema(name_, schema, debug, *j->inferred_function_schema, j->debug);
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
      TORCH_INTERNAL_ASSERT(*cpp_signature == *cpp_signature_,
        "Tried to register a kernel (", debug, ") for operator ", name_," for dispatch key ", toString(dispatch_key),
        ", but the C++ function signature ", cpp_signature->name(), " mismatched with a previous kernel that had the signature ",
        cpp_signature_->name()
      );
    } else {
      cpp_signature_ = *cpp_signature;
    }
  }

  if (schema_ && inferred_function_schema) {
    checkSchema(name_, schema_->schema, schema_->debug, *inferred_function_schema, debug);
  }

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : catchAllKernel_;

  if (k.size() > 0) {
    TORCH_WARN("Registering a kernel (", debug, ") for operator ", name_, " for dispatch key ", toString(dispatch_key), " that overwrote a previously registered kernel with the same dispatch key for the same operator.");
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
  if (dispatch_key.has_value()) {
    auto found = kernels_.find(*dispatch_key);
    TORCH_INTERNAL_ASSERT(found != kernels_.end(), "Tried to deregister a kernel for dispatch key ", toString(dispatch_key), " but there are no kernels registered for this dispatch key. The operator is ", toString(name_));
    auto& k = found->second;
    k.erase(kernel);
    if (k.empty()) {
      // the invariant says we don't want empty lists but instead remove the list from the map
      kernels_.erase(found);
    }
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    catchAllKernel_.erase(kernel);
    updateDispatchTableFull_(dispatcher);
  }
}

void OperatorEntry::updateFallback(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  updateDispatchTable_(dispatcher, dispatch_key);
}

KernelFunction OperatorEntry::computeDispatchTableEntry(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) const {
  auto dispatch_ix = static_cast<uint8_t>(dispatch_key);

  // 1. Operator registration
  auto kern_it = kernels_.find(dispatch_key);
  if (kern_it != kernels_.end()) {
    TORCH_INTERNAL_ASSERT(!kern_it->second.empty());
    return kern_it->second.front().kernel;

  // 2. Backend fallback
  } else if (dispatcher.backendFallbackKernels_[dispatch_ix].kernel.isValid()) {
    return dispatcher.backendFallbackKernels_[dispatch_ix].kernel;

  // 3. Catch all
  } else if (!catchAllKernel_.empty()) {
    return catchAllKernel_.front().kernel;

  // 4. Default to error
  } else {
    return KernelFunction();
  }
}

void OperatorEntry::updateDispatchTable_(const c10::Dispatcher& dispatcher, DispatchKey dispatch_key) {
  auto dispatch_ix = static_cast<uint8_t>(dispatch_key);
  dispatchTable_[dispatch_ix] = computeDispatchTableEntry(dispatcher, dispatch_key);
  dispatchKeyExtractor_.setOperatorHasFallthroughForBackend(dispatch_key, dispatchTable_[dispatch_ix].isFallthrough());
}

void OperatorEntry::updateDispatchTableFull_(const c10::Dispatcher& dispatcher) {
  for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
    updateDispatchTable_(dispatcher, static_cast<DispatchKey>(iter));
  }
}

void OperatorEntry::setManuallyBoxedKernel_(const c10::Dispatcher& dispatcher, KernelFunction::InternalBoxedKernelFunction* func) {
  TORCH_INTERNAL_ASSERT(!manuallyBoxedKernel_);
  manuallyBoxedKernel_ = func;

  for (auto& kv : kernels_) {
    for (auto& k : kv.second) {
      k.kernel.setManuallyBoxedKernel_(func);
    }
  }
  for (auto& k : catchAllKernel_) {
    k.kernel.setManuallyBoxedKernel_(func);
  }

  // Refresh entries in dispatchTable_
  updateDispatchTableFull_(dispatcher);
}

void OperatorEntry::checkInvariants() const {
  if (schema_) {
    TORCH_INTERNAL_ASSERT(schema_->schema.operator_name() == name_);
    dispatchKeyExtractor().checkInvariants(schema_->schema);
  }
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
  for (const auto& kv : kernels_) {
    TORCH_INTERNAL_ASSERT(kv.second.size() > 0);
  }
  for (uint8_t iter = 0; iter != static_cast<uint8_t>(DispatchKey::NumDispatchKeys); ++iter) {
    auto expected_k = computeDispatchTableEntry(c10::Dispatcher::singleton(), static_cast<DispatchKey>(iter));
    expected_k._equalsBoxedAndUnboxed(dispatchTable_[iter]);
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
  if (dispatchKey == DispatchKey::Undefined) {
    TORCH_CHECK(false,
          "There were no tensor arguments to this function (e.g., you passed an "
          "empty list of Tensors), but no fallback function is registered for schema ", name_,
          ".  This usually means that this function requires a non-empty list of Tensors.  "
          "Available functions are ", listAllDispatchKeys())
  }

  TORCH_CHECK(false, "Could not run '", name_, "' with arguments",
          " from the '", toString(dispatchKey), "' backend. '",
          name_, "' is only available for these backends: ",
          listAllDispatchKeys(), ".");
}

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
  // Iterate over DispatchKey, not the flat hash map, so we have a stable order
  auto print_kernel = [&](const char* k_desc, const std::list<AnnotatedKernel>& jts) {
    int64_t i = 0;
    for (const auto& jt : jts) {
      oss << k_desc
          << (i > 0 ? " (inactive)" : "")
          << ": "
          << jt.debug << " :: "
          << toString(*jt.inferred_function_schema) << " [ " << jt.kernel.dumpState() << "]\n";
      i++;
    }
  };

  for (uint8_t i = 0; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys); i++) {
    auto k = static_cast<DispatchKey>(i);
    auto it = kernels_.find(k);
    if (it != kernels_.end()) {
      print_kernel(toString(k), it->second);
    }
  }
  print_kernel("catchall", catchAllKernel_);
  // TODO: some invariants about manual boxing
  return oss.str();
}

}
}
