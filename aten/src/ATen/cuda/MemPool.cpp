#include <ATen/cuda/MemPool.h>

namespace at::cuda {

MemPool::MemPool(
    CUDACachingAllocator::CUDAAllocator* allocator,
    bool is_user_created,
    bool use_on_oom,
    bool no_split)
    : allocator_(allocator),
      is_user_created_(is_user_created),
      id_(c10::generate_mempool_id(is_user_created)) {
  device_ = c10::cuda::current_device();
  CUDACachingAllocator::createOrIncrefPool(device_, id_, allocator);
  if (use_on_oom) {
    CUDACachingAllocator::setUseOnOOM(device_, id_, true);
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
  CUDACachingAllocator::setUseOnOOM(device_, id_, false);
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
  return c10::generate_mempool_id(is_user_created);
}

} // namespace at::cuda
