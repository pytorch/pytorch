// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLCCA.hpp>

#include <ATen/Context.h>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

namespace c10d::nccl2 {

namespace {
void ncclCachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  NcclCachingAllocatorHook::getInstance().regDeregMem(te);
}
} // namespace

NcclCachingAllocatorHook& NcclCachingAllocatorHook::getInstance() {
  // Leaked singleton: allocator trace trackers cannot be detached, so the
  // hook must outlive every allocator event.
  static auto* instance = new NcclCachingAllocatorHook();
  return *instance;
}

NcclCachingAllocatorHook::NcclCachingAllocatorHook() {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  registerMemPreHook();
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &ncclCachingAllocatorHookFn);
}

void NcclCachingAllocatorHook::registerMemPreHook() {
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segmentInfo : snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    registeredMemMap_.emplace(
        addr, MemInfo{segmentInfo.total_size, segmentInfo.device});
  }
}

void NcclCachingAllocatorHook::regDeregMem(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;
    TORCH_CHECK(
        !registeredMemMap_.count(addr), "Memory already registered with NCCL");
    registeredMemMap_.emplace(addr, MemInfo{len, te.device_});
    for (auto* comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->register_address(addr, len);
      }
    }
  } else if (
      te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    TORCH_CHECK(
        registeredMemMap_.count(addr), "Memory not registered with NCCL");
    registeredMemMap_.erase(addr);
    for (auto* comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->deregister_address(addr);
      }
    }
  }
}

void NcclCachingAllocatorHook::registerComm(ProcessGroupNCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!registeredComms_.count(comm), "Communicator already registered");
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->register_address(addr, mem_info.len);
    }
  }
  registeredComms_.insert(comm);
}

void NcclCachingAllocatorHook::deregisterComm(ProcessGroupNCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!registeredComms_.count(comm)) {
    return;
  }
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->deregister_address(addr);
    }
  }
  registeredComms_.erase(comm);
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
