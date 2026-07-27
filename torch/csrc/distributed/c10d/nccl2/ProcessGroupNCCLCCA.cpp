// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCLCCA.hpp>

#include <ATen/Context.h>
#include <c10/core/AllocatorConfig.h>
#include <torch/csrc/distributed/c10d/nccl2/Logging.hpp>
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

NcclCachingAllocatorHook::NcclCachingAllocatorHook()
    : register_default_pool_segments_(
          !c10::CachingAllocator::AcceleratorAllocatorConfig::
              use_expandable_segments()) {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  if (!register_default_pool_segments_) {
    TC_LOG(INFO)
        << "Disabling default-pool NCCL memory registration because it is "
           "incompatible with CUDA allocator expandable segments mode.";
  }
  registerMemPreHook();
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &ncclCachingAllocatorHookFn);
}

bool NcclCachingAllocatorHook::shouldTrackSegment(
    const c10::MempoolId_t& mempool_id) const {
  return register_default_pool_segments_ ||
      mempool_id != c10::MempoolId_t{0, 0};
}

void NcclCachingAllocatorHook::registerMemPreHook() {
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segmentInfo : snapshot.segments) {
    if (!shouldTrackSegment(segmentInfo.owner_private_pool_id)) {
      continue;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    registeredMemMap_.emplace(
        addr, MemInfo{segmentInfo.total_size, segmentInfo.device});
  }
}

void NcclCachingAllocatorHook::regDeregMem(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  if (!shouldTrackSegment(te.mempool_)) {
    return;
  }
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
