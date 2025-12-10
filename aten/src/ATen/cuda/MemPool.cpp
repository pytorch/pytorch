#include <ATen/cuda/MemPool.h>

namespace at::cuda {

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
    bool is_user_created,
    bool use_on_oom,
    bool no_split)
    : allocator_(allocator), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10::cuda::current_device();
  CUDACachingAllocator::createOrIncrefPool(device_, id_, allocator);
  if (use_on_oom) {
    CUDACachingAllocator::setUseOnOOM(device_, id_);
  }
  if (no_split) {
    CUDACachingAllocator::setNoSplit(device_, id_);
  }
}

MemPool::~MemPool() {
  // TORCH_INTERNAL_ASSERT(use_count() == 1);
  // We used to assert that TORCH_INTERNAL_ASSERT(use_count() == 1);
  // However, this assertion is not true if a memory pool is shared
  // with a cuda graph. That CUDAGraph will increase the use count
  // until it is reset.
  CUDACachingAllocator::releasePool(device_, id_);
  c10::cuda::CUDACachingAllocator::emptyCache(id_);
}

MempoolId_t MemPool::id() {
  return id_;
}

CUDACachingAllocator::CUDAAllocator* MemPool::allocator() {
  return allocator_;
}

int MemPool::use_count() {
  return CUDACachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device() {
  return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

} // namespace at::cuda
