#include <ATen/core/MemPool.h>

#include <atomic>

namespace at {

namespace {

// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10::cuda::MemPool().
//
// uuid_ is incremented when accelerator::Graph/CUDAGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or accelerator::Graph/CUDAGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> uid_{1};
std::atomic<CaptureId_t> uuid_{1};

} // anonymous namespace

c10::MempoolId_t create_mempool_id(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

} // namespace at
