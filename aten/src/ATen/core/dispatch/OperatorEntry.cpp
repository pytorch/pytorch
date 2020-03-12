#include <ATen/core/dispatch/OperatorEntry.h>

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

  std::string listAllDispatchKeys(const ska::flat_hash_map<c10::optional<DispatchKey>, std::list<OperatorEntry::ListEntry>>& kernels) {
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
  schema_ = std::move(schema);
  dispatchTable_.registerSchema(*schema_);
  for (auto i = kernels_.begin(); i != kernels_.end(); ++i) {
    for (auto j = i->second.begin(); j != i->second.end(); ++j) {
      if (j->inferred_function_schema) {
        checkSchema(name_, *schema_, *j->inferred_function_schema);
      }
    }
  }
}

void OperatorEntry::deregisterSchema() {
  TORCH_INTERNAL_ASSERT(schema_.has_value());
  schema_ = c10::nullopt;
  dispatchTable_.deregisterSchema();
}

std::list<OperatorEntry::ListEntry>::iterator OperatorEntry::registerKernel(c10::optional<DispatchKey> dispatch_key, KernelFunction kernel, std::unique_ptr<FunctionSchema> inferred_function_schema) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  if (schema_ && inferred_function_schema) {
    checkSchema(name_, *schema_, *inferred_function_schema);
  }

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  auto& k = kernels_[dispatch_key];
  k.emplace_front(std::move(kernel), std::move(inferred_function_schema));
  std::list<OperatorEntry::ListEntry>::iterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  updateDispatchTable_(dispatch_key);
  return inserted;
}

void OperatorEntry::deregisterKernel_(c10::optional<DispatchKey> dispatch_key, std::list<OperatorEntry::ListEntry>::iterator kernel) {
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

std::string OperatorEntry::dumpState() const {
  std::ostringstream oss;
  oss << "name: " << name_ << "\n";
  if (schema_) {
    oss << "schema: " << *schema_ << "\n";
  } else {
    oss << "schema: (none)\n";
  }
  return oss.str();
}

}
}
