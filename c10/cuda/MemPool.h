#pragma once

#include <atomic>
#include <functional>

#include <c10/cuda/CUDAMacros.h>

namespace c10::cuda {

using CaptureId_t = unsigned long long;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;
using AllocFuncSignature = int(void**, size_t);
using DeleteFuncSignature = int(void*);
using AllocFuncType = std::function<AllocFuncSignature>;
using DeleteFuncType = std::function<DeleteFuncSignature>;

struct C10_CUDA_API MemPool {
  MemPool(uint64_t alloc_fn=0, uint64_t delete_fn=0, bool is_user_created=true);

  MempoolId_t id_; 
  AllocFuncType alloc_fn_;
  DeleteFuncType delete_fn_;
  bool is_user_created_;
  
  private:
    static std::atomic<CaptureId_t> uid_;
    static std::atomic<CaptureId_t> uuid_;
 
};

struct C10_CUDA_API MemPoolContext {
  MemPoolContext(MemPool* mempool);

  ~MemPoolContext();
  
  static MemPool* getActiveMemPool();
  static void setActiveMemPool(MemPool* mempool);
  
  private:
    MemPool* prev_mempool_;
    static thread_local MemPool* active_mempool_;
};

}
