#include <ATen/core/dispatch/OperatorEntry.h>

namespace c10 {
namespace impl {

namespace {
  std::string listAllDispatchKeys(const ska::flat_hash_map<TensorTypeId, std::list<DispatchTableEntry>>& kernels) {
    if (kernels.size() == 0) {
      return "";
    }
    std::ostringstream str;
    str << detail::dispatch_key_to_string(kernels.begin()->first);
    for (auto iter = ++kernels.begin(); iter != kernels.end(); ++iter) {
      str << ", " << detail::dispatch_key_to_string(iter->first);
    }
    return str.str();
  }
}

OperatorEntry::OperatorEntry(FunctionSchema&& schema)
: schema_(std::move(schema))
, dispatchTable_(schema_)
, kernels_(make_left<ska::flat_hash_map<TensorTypeId, std::list<DispatchTableEntry>>, std::list<DispatchTableEntry>>()) {
}

void OperatorEntry::prepareForDeregistration() {
  return dispatchTable_.read([&] (const DispatchTable& dispatchTable) {
    if (!dispatchTable.isEmpty()) {
      AT_ERROR("Tried to deregister op schema for an operator that still has kernels registered. The operator schema is ", toString(schema_), ". Registered kernels for dispatch keys: ", dispatchTable.listAllDispatchKeys());
    }
  });
  AT_ASSERTM(kernels_.is_left(), "If the dispatch table is empty, then the invariant says there can't be any kernels but we still have a catch-all kernel. The operator schema is ", toString(schema_));
  AT_ASSERTM(kernels_.left().size() == 0, "If the dispatch table is empty, then the invariant says there can't be any kernels but we still have kernels for dispatch keys ", listAllDispatchKeys(kernels_.left()), ". The operator schema is ", toString(schema_));
}

RegistrationHandleRAII OperatorEntry::registerKernel(TensorTypeId dispatch_key, DispatchTableEntry kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  AT_CHECK(kernels_.is_left(), "Tried to register a kernel with dispatch key ", detail::dispatch_key_to_string(dispatch_key)," for an operator which already has a catch-all kernel registered. An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is ", toString(schema_));

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  auto& k = kernels_.left()[dispatch_key];
  k.push_front(kernel);
  std::list<DispatchTableEntry>::iterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  updateDispatchTable_(dispatch_key);

  return RegistrationHandleRAII([this, dispatch_key, inserted] {
    // list iterators stay valid even if the list changes,
    // so we can use the iterator to remove the kernel from the list
    deregisterKernel_(dispatch_key, inserted);
  });
}

RegistrationHandleRAII OperatorEntry::registerCatchallKernel(DispatchTableEntry kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  if (kernels_.is_left()) {
    AT_CHECK(0 == kernels_.left().size(), "Tried to register a catch-all kernel for an operator which already has kernels for dispatch keys ", listAllDispatchKeys(kernels_.left()), ". An operator can only have either a catch-all kernel or kernels with dispatch keys. The operator schema is ", toString(schema_));
    kernels_ = make_right<ska::flat_hash_map<TensorTypeId, std::list<DispatchTableEntry>>, std::list<DispatchTableEntry>>();
  }

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  auto& k = kernels_.right();
  k.push_front(kernel);
  std::list<DispatchTableEntry>::iterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  updateCatchallDispatchTable_();

  return RegistrationHandleRAII([this, inserted] {
    // list iterators stay valid even if the list changes,
    // so we can use the iterator to deregister the kernel from the list
    deregisterCatchallKernel_(inserted);
  });
}

void OperatorEntry::deregisterKernel_(TensorTypeId dispatch_key, std::list<DispatchTableEntry>::iterator kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  AT_CHECK(kernels_.is_left(), "Tried deregister a kernel for dispatch key ", detail::dispatch_key_to_string(dispatch_key), " for an operator that only has a catch-all kernel. The operator schema is ", toString(schema_));

  auto& kernels = kernels_.left();
  auto found = kernels.find(dispatch_key);
  AT_ASSERTM(found != kernels.end(), "Tried to deregister a kernel for dispatch key ", detail::dispatch_key_to_string(dispatch_key), " but there are no kernels registered for this dispatch key. The operator schema is ", toString(schema_));
  auto& k = found->second;
  k.erase(kernel);
  if (k.empty()) {
    // the invariant says we don't want empty lists but instead remove the list from the map
    kernels.erase(found);
  }

  updateDispatchTable_(dispatch_key);
}

void OperatorEntry::deregisterCatchallKernel_(std::list<DispatchTableEntry>::iterator kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  AT_CHECK(kernels_.is_right(), "Tried to deregister a catch-all kernel for an operator that doesn't have a catch-all kernel registered. The operator schema is ", toString(schema_));

  auto& k = kernels_.right();
  k.erase(kernel);
  if (k.empty()) {
    // the invariant says that the empty state is represented with is_left()
    kernels_ = make_left<ska::flat_hash_map<TensorTypeId, std::list<DispatchTableEntry>>, std::list<DispatchTableEntry>>();
  }

  updateCatchallDispatchTable_();
}

void OperatorEntry::updateDispatchTable_(TensorTypeId dispatch_key) {
  // precondition: kernelsMutex_ is locked

  AT_ASSERTM(kernels_.is_left(), "Can't update the dispatch table a dispatch key ", detail::dispatch_key_to_string(dispatch_key), " because the operator only has catch-all kernels. The operator schema is ", toString(schema_));

  auto& kernels = kernels_.left();
  auto k = kernels.find(dispatch_key);

  if (k == kernels.end()) {
    dispatchTable_.write([&] (DispatchTable& dispatchTable) {
      dispatchTable.removeKernelIfExists(dispatch_key);
    });
  } else {
    dispatchTable_.write([&] (DispatchTable& dispatchTable) {
      dispatchTable.setKernel(dispatch_key, k->second.front());
    });
  }
}

void OperatorEntry::updateCatchallDispatchTable_() {
  // precondition: kernelsMutex_ is locked

  if (kernels_.is_left()) {
    dispatchTable_.write([&] (DispatchTable& dispatchTable) {
      dispatchTable.removeCatchallKernel();
    });
  } else {
    dispatchTable_.write([&] (DispatchTable& dispatchTable) {
      dispatchTable.setCatchallKernel(kernels_.right().front());
    });
  }
}

}
}
