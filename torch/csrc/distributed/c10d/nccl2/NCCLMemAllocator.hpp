// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Allocator behind torch.cuda.MemPool(backend.allocator(symm)). Its segments
// come from ncclMemAlloc, and NCCLCachingAllocatorHook registers them with
// every process group like any other segment. Symmetric windows are collective,
// so they are handled here: the caching allocator calls raw_alloc/raw_delete
// without holding its mutex. A symm allocator gives each segment a window on
// its process group when allocated, and before ncclMemFree every allocator
// removes the segment's windows from the attached process groups.

#pragma once

#ifdef USE_C10D_NCCL

#include <memory>
#include <mutex>
#include <vector>

#include <torch/csrc/cuda/CUDAPluggableAllocator.h>

namespace c10d::nccl2 {

class NcclApi;
class ProcessGroupNCCL;

class TORCH_API NCCLMemAllocator final
    : public torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator {
 public:
  // The process-wide symm=false allocator (backend.mem_allocator). Every
  // process group attaches to it, so freeing a segment removes the windows
  // that new_window gave it on any process group.
  static const std::shared_ptr<NCCLMemAllocator>& global();

  // symm gives every segment a collective symmetric window on the attached
  // process group, so every rank must allocate the pool's segments in the same
  // order and sizes, and free them together.
  explicit NCCLMemAllocator(bool symm);

  bool symm() const {
    return symm_;
  }
  // Whether ptr is the base of a live segment of this allocator.
  bool ownsSegment(const void* ptr) const;
  // Removes pg's windows on a segment when the segment is freed. With symm,
  // also gives the live and future segments windows on pg.
  void attach(ProcessGroupNCCL& pg);
  // Removes pg's windows on the live segments and stops tracking pg.
  void detach(ProcessGroupNCCL& pg);

 private:
  struct Segment {
    void* ptr;
    size_t size;
    c10::DeviceIndex device;
  };

  void* allocSegment(size_t size, int device);
  void freeSegment(void* ptr, int device);

  std::shared_ptr<NcclApi> nccl_api_;
  const bool symm_;
  // Held across the window calls, which take the process group's
  // memory_registration_mutex_; ProcessGroupNCCL must not call attach or
  // detach while holding that mutex.
  std::mutex mutex_;
  std::vector<ProcessGroupNCCL*> pgs_;
  // Written with both mutexes held; ownsSegment reads it with only this one,
  // so it can run under memory_registration_mutex_.
  mutable std::mutex segments_mutex_;
  // In allocation order, which is the cross-rank order for collective window
  // calls.
  std::vector<Segment> segments_;
};

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
