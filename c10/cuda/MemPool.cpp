#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/MemPool.h>

namespace c10::cuda {

// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10::cuda::MemPool().
//
// uuid_ is incremented when CUDAGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or CUDAGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    CUDACachingAllocator::CUDAAllocator* allocator,
    bool is_user_created)
    : allocator_(allocator), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
}

MempoolId_t MemPool::id() {
  return id_;
}

CUDACachingAllocator::CUDAAllocator* MemPool::allocator() {
  return allocator_;
}

thread_local MemPool* MemPoolContext::active_mempool_ = nullptr;

MemPoolContext::MemPoolContext(MemPool* mempool)
    : prev_mempool_(active_mempool_) {
  active_mempool_ = mempool;
}

MemPoolContext::~MemPoolContext() {
  active_mempool_ = prev_mempool_;
}

MemPool* MemPoolContext::getActiveMemPool() {
  return active_mempool_;
}

} // namespace c10::cuda
