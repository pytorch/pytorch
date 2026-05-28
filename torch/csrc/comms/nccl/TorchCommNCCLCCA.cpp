// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/nccl/TorchCommNCCLCCA.hpp>

namespace torch::comms {

// Global function to be registered as a hook
void ncclCachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // Forward to the singleton instance
  NcclCachingAllocatorHook::getInstance().regDeregMem(te);
}

NcclCachingAllocatorHookImpl& NcclCachingAllocatorHook::getInstance() {
  // Use c10::call_once for thread-safe singleton initialization
  // NOLINTNEXTLINE(facebook-hte-c10::call_once)
  c10::call_once(init_flag_, createInstance);
  return *instance_;
}

DefaultNcclCachingAllocatorHookImpl::DefaultNcclCachingAllocatorHookImpl() {
  // Setup memory registration hooks
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  registerMemPreHook();
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &ncclCachingAllocatorHookFn);
}

void NcclCachingAllocatorHookImpl::registerMemPreHook() {
  // We assume no mem pool and no comm has been created yet, we just loop up the
  // snapshot of the default pool for all devices.
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segmentInfo : snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    size_t len = segmentInfo.total_size;

    if (registeredMemMap_.contains(addr)) {
      throw std::runtime_error("Memory already registered with NCCL");
    } else {
      registeredMemMap_.emplace(addr, MemInfo{len, segmentInfo.device});
    }
  }
}

void NcclCachingAllocatorHookImpl::regDeregMem(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool register_mem = te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC;
  bool unregister_mem = te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE;

  if (register_mem) {
    // Memory got allocated, register it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (registeredMemMap_.contains(addr)) {
      throw std::runtime_error("Memory already registered with NCCL");
    } else {
      registeredMemMap_.emplace(addr, MemInfo{len, te.device_});
    }

    // Register the memory through ncclCommRegister and add to commRegHandles_
    for (auto& comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->register_address(TorchCommNCCL::AddressWithLen(addr, len));
      }
    }
  } else if (unregister_mem) {
    // Memory got freed, deregister it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));

    if (!registeredMemMap_.contains(addr)) {
      throw std::runtime_error("Memory not registered with NCCL");
    } else {
      registeredMemMap_.erase(addr);
    }

    for (auto& comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->deregister_address(TorchCommNCCL::Address(addr));
      }
    }
  }
}

void NcclCachingAllocatorHookImpl::registerComm(TorchCommNCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the communicator is already registered
  if (registeredComms_.contains(comm)) {
    throw std::runtime_error("Communicator already registered");
  }

  // Register all memory that has already been allocated
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->register_address(TorchCommNCCL::AddressWithLen(addr, mem_info.len));
    }
  }

  registeredComms_.insert(comm);
}

void NcclCachingAllocatorHookImpl::deregisterComm(TorchCommNCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!registeredComms_.contains(comm)) {
    // Should this be fatal?
    return;
  }

  // De-register all memory that has already been allocated
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->deregister_address(TorchCommNCCL::Address(addr));
    }
  }

  registeredComms_.erase(comm);
}

void NcclCachingAllocatorHookImpl::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& comm : registeredComms_) {
    for (const auto& [addr, mem_info] : registeredMemMap_) {
      if (mem_info.device == comm->getDevice().index()) {
        comm->deregister_address(TorchCommNCCL::Address(addr));
      }
    }
  }
  registeredMemMap_.clear();
  registeredComms_.clear();
}

bool NcclCachingAllocatorHookImpl::isCommRegistered(TorchCommNCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  return registeredComms_.contains(comm);
}

} // namespace torch::comms
