#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/op_registration/infer_schema.h>

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

  std::string listAllDispatchKeys(const ska::flat_hash_map<c10::optional<DispatchKey>, std::list<OperatorEntry::KernelEntry>>& kernels) {
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

OperatorEntry::OperatorEntry(FunctionSchema&& schema)
: name_(schema.operator_name())
, schema_(std::move(schema))
, dispatchTable_(*schema_)
, kernels_() {
}

OperatorEntry::OperatorEntry(OperatorName&& operator_name)
: name_(std::move(operator_name))
, schema_()
, dispatchTable_(name_)
, kernels_() {
}

void OperatorEntry::prepareForDeregistration() {
  if (!dispatchTable_.isEmpty()) {
     TORCH_INTERNAL_ASSERT(false, "Tried to deregister op schema for an operator that still has kernels registered. The operator is ", toString(name_), ". Registered kernels for dispatch keys: ", dispatchTable_.listAllDispatchKeys());
  }
  TORCH_INTERNAL_ASSERT(kernels_.size() == 0, "If the dispatch table is empty, then the invariant says there can't be any kernels but we still have kernels for dispatch keys ", listAllDispatchKeys(kernels_), ". The operator is ", toString(name_));
}

namespace {
  void checkSchema(const OperatorName& name, const FunctionSchema& from_def, const FunctionSchema& inferred) {
    c10::optional<std::string> schema_difference = findSchemaDifferences(from_def, inferred);
    if (schema_difference.has_value()) {
      TORCH_CHECK(false,
        "In registration for ", toString(name), ": expected schema of operator to be \"", toString(from_def), "\", ",
        "but got inferred schema \"", toString(inferred), "\". ",
        *schema_difference);
    }
  }
}

void OperatorEntry::registerSchema(FunctionSchema&& schema) {
  TORCH_INTERNAL_ASSERT(!schema_.has_value());
  for (auto i = kernels_.begin(); i != kernels_.end(); ++i) {
    for (auto j = i->second.begin(); j != i->second.end(); ++j) {
      if (j->inferred_function_schema != nullptr) {
        checkSchema(name_, schema, *j->inferred_function_schema);
      }
    }
  }
  // NB: don't register schema until after we've checked everything!
  schema_ = std::move(schema);
  dispatchTable_.registerSchema(*schema_);
}

void OperatorEntry::deregisterSchema() {
  TORCH_INTERNAL_ASSERT(schema_.has_value());
  schema_ = c10::nullopt;
  dispatchTable_.deregisterSchema();
}

std::list<OperatorEntry::KernelEntry>::iterator OperatorEntry::registerKernel(
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  if (schema_ && inferred_function_schema) {
    checkSchema(name_, *schema_, *inferred_function_schema);
  }

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  auto& k = kernels_[dispatch_key];

  if (k.size() > 0) {
    TORCH_WARN("Registering a kernel (", debug, ") for operator ", name_, " for dispatch key ", toString(dispatch_key), " that overwrote a previously registered kernel with the same dispatch key for the same operator.");
  }

  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
  std::list<OperatorEntry::KernelEntry>::iterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  updateDispatchTable_(dispatch_key);
  return inserted;
}

void OperatorEntry::deregisterKernel_(
  c10::optional<DispatchKey> dispatch_key,
  std::list<OperatorEntry::KernelEntry>::iterator kernel
) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  auto found = kernels_.find(dispatch_key);
  TORCH_INTERNAL_ASSERT(found != kernels_.end(), "Tried to deregister a kernel for dispatch key ", toString(dispatch_key), " but there are no kernels registered for this dispatch key. The operator is ", toString(name_));
  auto& k = found->second;
  k.erase(kernel);
  if (k.empty()) {
    // the invariant says we don't want empty lists but instead remove the list from the map
    kernels_.erase(found);
  }

  updateDispatchTable_(dispatch_key);
}

void OperatorEntry::updateDispatchTable_(c10::optional<DispatchKey> dispatch_key) {
  // precondition: kernelsMutex_ is locked

  auto k = kernels_.find(dispatch_key);
  if (dispatch_key.has_value()) {
    if (k == kernels_.end()) {
      dispatchTable_.removeKernelIfExists(*dispatch_key);
    } else {
      dispatchTable_.setKernel(*dispatch_key, k->second.front().kernel);
    }
  } else {
    if (k == kernels_.end()) {
      dispatchTable_.removeCatchallKernel();
    } else {
      dispatchTable_.setCatchallKernel(k->second.front().kernel);
    }
  }
}

void OperatorEntry::checkInvariants() const {
  if (schema_) {
    TORCH_INTERNAL_ASSERT(schema_->operator_name() == name_);
    dispatchTable_.dispatchKeyExtractor().checkInvariants(*schema_);
  }
  TORCH_INTERNAL_ASSERT(name_ == dispatchTable_.operatorName());
  TORCH_INTERNAL_ASSERT(kernels_.find(DispatchKey::Undefined) == kernels_.end());
  for (const auto& kv : kernels_) {
    auto mb_dispatch_key = kv.first;
    TORCH_INTERNAL_ASSERT(kv.second.size() > 0);
    auto* kernel = mb_dispatch_key ? dispatchTable_.lookup(*mb_dispatch_key) : dispatchTable_.lookupCatchallKernel();
    auto manual_boxed_kernel = dispatchTable_.manuallyBoxedKernel();
    // NB: this is a copy
    auto local_kernel = kv.second.front().kernel;
    if (manual_boxed_kernel.has_value()) {
      local_kernel.setManuallyBoxedKernel_(*manual_boxed_kernel);
    }
    TORCH_INTERNAL_ASSERT(local_kernel._equalsBoxedAndUnboxed(*kernel));
  }
}

std::string OperatorEntry::dumpState() const {
  std::ostringstream oss;
  oss << "name: " << name_ << "\n";
  if (schema_) {
    oss << "schema: " << *schema_ << "\n";
    oss << "alias analysis kind: " << toString(schema_->aliasAnalysis()) << (schema_->isDefaultAliasAnalysisKind() ? " (default)" : "") << "\n";
  } else {
    oss << "schema: (none)\n";
  }
  // Iterate over DispatchKey, not the flat hash map, so we have a stable order
  auto print_key = [&](c10::optional<DispatchKey> k) {
    auto it = kernels_.find(k);
    if (it != kernels_.end()) {
      int64_t i = 0;
      for (const auto& jt : it->second) {
        oss << (k ? toString(k) : "catchall")
            << (i > 0 ? " (inactive)" : "")
            << ": "
            << jt.debug << " :: "
            << toString(*jt.inferred_function_schema) << " [ " << jt.kernel.dumpState() << "]\n";
        i++;
      }
    }
  };
  for (uint8_t i = 0; i < static_cast<uint8_t>(DispatchKey::NumDispatchKeys); i++) {
    print_key(static_cast<DispatchKey>(i));
  }
  print_key(c10::nullopt);
  // dispatch table is 100% specified by OperatorEntry; so if you want to check
  // if it makes sense use checkInvariants
  // oss << dispatchTable_.dumpState();
  return oss.str();
}

}
}
