#pragma once

#include <atomic>
#include <functional>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAMacros.h>

namespace c10::cuda {

using CaptureId_t = unsigned long long;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

// forward declare CUDAAllocator
namespace CUDACachingAllocator {
class CUDAAllocator;
} // namespace CUDACachingAllocator

struct C10_CUDA_API MemPool {
  MemPool(
      CUDACachingAllocator::CUDAAllocator* allocator = nullptr,
      bool is_user_created = true);

  MempoolId_t id();

  CUDACachingAllocator::CUDAAllocator* allocator_;
  bool is_user_created_;

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  MempoolId_t id_;
};

struct C10_CUDA_API MemPoolContext {
  MemPoolContext(MemPool* mempool);

  ~MemPoolContext();

  static MemPool* getActiveMemPool();

 private:
  MemPool* prev_mempool_;
  static thread_local MemPool* active_mempool_;
};

} // namespace c10::cuda
