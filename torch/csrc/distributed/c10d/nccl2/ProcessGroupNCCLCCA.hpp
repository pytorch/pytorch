// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// CUDA caching allocator hook for the nccl2 backend. Port of torchcomms'
// NcclCachingAllocatorHook: a process-wide singleton that watches allocator
// SEGMENT_ALLOC/SEGMENT_FREE trace events and forwards them to every
// registered ProcessGroupNCCL as register_address/deregister_address, so
// segments from the NCCL mempool are registered with each communicator
// (ncclCommRegister eagerly; the collective NCCL_WIN_COLL_SYMMETRIC window
// registration happens lazily via ensureSegmentWindow).

#pragma once

#ifdef USE_C10D_NCCL

#include <mutex>
#include <set>
#include <unordered_map>

#include <c10/cuda/CUDACachingAllocator.h>

namespace c10d::nccl2 {

class ProcessGroupNCCL;

class NcclCachingAllocatorHook {
 public:
  static NcclCachingAllocatorHook& getInstance();

  void regDeregMem(const c10::cuda::CUDACachingAllocator::TraceEntry& te);
  void registerComm(ProcessGroupNCCL* comm);
  void deregisterComm(ProcessGroupNCCL* comm);

 private:
  NcclCachingAllocatorHook();

  // Seed registeredMemMap_ with segments that existed before the hook was
  // attached, so a comm registered later still sees them.
  void registerMemPreHook();
  bool shouldTrackSegment(const c10::MempoolId_t& mempool_id) const;

  struct MemInfo {
    size_t len;
    int32_t device;
  };

  std::mutex mutex_;
  std::unordered_map<void*, MemInfo> registeredMemMap_;
  std::set<ProcessGroupNCCL*> registeredComms_;
  bool register_default_pool_segments_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
