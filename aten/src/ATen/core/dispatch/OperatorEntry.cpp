#include <ATen/core/dispatch/OperatorEntry.h>

namespace c10 {
namespace impl {

namespace {
  std::string listAllDispatchKeys(const ska::flat_hash_map<TensorTypeId, std::list<KernelFunction>>& kernels) {
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

OperatorEntry::OperatorEntry(FunctionSchema&& schema, OperatorOptions&& options)
: schema_(std::move(schema))
, dispatchTable_(schema_)
, kernels_()
, catchAllKernels_()
, options_(std::move(options)) {
}

void OperatorEntry::prepareForDeregistration() {
  return dispatchTable_.read([&] (const DispatchTable& dispatchTable) {
    if (!dispatchTable.isEmpty()) {
      TORCH_INTERNAL_ASSERT(false, "Tried to deregister op schema for an operator that still has kernels registered. The operator schema is ", toString(schema_), ". Registered kernels for dispatch keys: ", dispatchTable.listAllDispatchKeys());
    }
  });
  TORCH_INTERNAL_ASSERT(kernels_.size() == 0, "If the dispatch table is empty, then the invariant says there can't be any kernels but we still have kernels for dispatch keys ", listAllDispatchKeys(kernels_), ". The operator schema is ", toString(schema_));
  TORCH_INTERNAL_ASSERT(catchAllKernels_.size() == 0, "If the dispatch table is empty, then the invariant says there can't be any kernels but we still have catch-all kernel. The operator schema is ", toString(schema_));
}

RegistrationHandleRAII OperatorEntry::registerKernel(TensorTypeId dispatch_key, KernelFunction kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  auto& k = kernels_[dispatch_key];
  k.push_front(kernel);
  std::list<KernelFunction>::iterator inserted = k.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  updateDispatchTable_(dispatch_key);

  return RegistrationHandleRAII([this, dispatch_key, inserted] {
    // list iterators stay valid even if the list changes,
    // so we can use the iterator to remove the kernel from the list
    deregisterKernel_(dispatch_key, inserted);
  });
}

RegistrationHandleRAII OperatorEntry::registerCatchallKernel(KernelFunction kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  // Add the kernel to the kernels list,
  // possibly creating the list if this is the first kernel.
  catchAllKernels_.push_front(kernel);
  std::list<KernelFunction>::iterator inserted = catchAllKernels_.begin();
  // update the dispatch table, i.e. re-establish the invariant
  // that the dispatch table points to the newest kernel
  updateCatchallDispatchTable_();

  return RegistrationHandleRAII([this, inserted] {
    // list iterators stay valid even if the list changes,
    // so we can use the iterator to deregister the kernel from the list
    deregisterCatchallKernel_(inserted);
  });
}

void OperatorEntry::deregisterKernel_(TensorTypeId dispatch_key, std::list<KernelFunction>::iterator kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  auto found = kernels_.find(dispatch_key);
  TORCH_INTERNAL_ASSERT(found != kernels_.end(), "Tried to deregister a kernel for dispatch key ", toString(dispatch_key), " but there are no kernels registered for this dispatch key. The operator schema is ", toString(schema_));
  auto& k = found->second;
  k.erase(kernel);
  if (k.empty()) {
    // the invariant says we don't want empty lists but instead remove the list from the map
    kernels_.erase(found);
  }

  updateDispatchTable_(dispatch_key);
}

void OperatorEntry::deregisterCatchallKernel_(std::list<KernelFunction>::iterator kernel) {
  std::unique_lock<std::mutex> lock(kernelsMutex_);

  catchAllKernels_.erase(kernel);

  updateCatchallDispatchTable_();
}

void OperatorEntry::updateDispatchTable_(TensorTypeId dispatch_key) {
  // precondition: kernelsMutex_ is locked

  auto k = kernels_.find(dispatch_key);

  if (k == kernels_.end()) {
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

  if (catchAllKernels_.size() == 0) {
    dispatchTable_.write([&] (DispatchTable& dispatchTable) {
      dispatchTable.removeCatchallKernel();
    });
  } else {
    dispatchTable_.write([&] (DispatchTable& dispatchTable) {
      dispatchTable.setCatchallKernel(catchAllKernels_.front());
    });
  }
}

}
}
