#include <ATen/xpu/MemPool.h>

namespace at::xpu {

// uid_ is incremented when a user creates a MemPool,
//
// uuid_ is incremented when XPUGraph creates a MemPool
// as a result of a user not providing a pool.

std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    XPUCachingAllocator::XPUAllocator* allocator,
    bool is_user_created,
    bool use_on_oom)
    : allocator_(allocator), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10::xpu::current_device();
  XPUCachingAllocator::createOrIncrefPool(device_, id_, allocator);
  if (use_on_oom) {
    // XPU doesn't support use_on_oom yet
    TORCH_WARN(
        "XPUCachingAllocator::MemPool: use_on_oom is not supported on XPU");
  }
}

MemPool::~MemPool() {
  // No use_count() == 1 assertion: a pool shared with an XPUGraph has
  // use_count > 1 until the graph is reset.
  XPUCachingAllocator::releasePool(device_, id_);
  c10::xpu::XPUCachingAllocator::emptyCache(id_); // release cached blocks
}

MempoolId_t MemPool::id() {
  return id_;
}

XPUCachingAllocator::XPUAllocator* MemPool::allocator() {
  return allocator_;
}

int MemPool::use_count() {
  return XPUCachingAllocator::getPoolUseCount(device_, id_);
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

} // namespace at::xpu
