#pragma once
#ifdef USE_CUDA
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Logging.h>
#include <cuda_runtime_api.h>
#include <ATen/detail/IPCShared.h>
#include <torch/csrc/Export.h>
#include <cstddef>
namespace torch {

using CudaCounterOps = at::ipc::AtomicCounterOps;

TORCH_CUDA_CU_API bool CudaIPCCollect();

struct CudaIPCReceivedData final {
  CudaIPCReceivedData() = default;
  explicit CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct CudaIPCSentData final
    : public at::ipc::SentDataBase<int64_t, CudaCounterOps> {
  cudaEvent_t event_; // Sync cuEventDestroy
  bool event_sync_required_;
  at::Device device_;

  CudaIPCSentData(
      std::string handle,
      uint64_t offset,
      int64_t* counter_ptr,
      at::Device device);
  ~CudaIPCSentData();

  int64_t counter_value();
};

TORCH_CUDA_CU_API at::DataPtr GetNewRefCountedSentData(
    void* data,
    at::Device device);

namespace {

inline constexpr int64_t CUDA_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;
// This was determined empirically that CUDA (v10.1 and below) have the limit
// on the number of recorded blocking interprocess events. It is around ~22,000.
// And to give us leeway, we picked 1000 as it gives us enough events to share
// tensors effectively.
inline constexpr int64_t CUDA_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

// All to be deleted data blocks with non zero reference counter goes there
struct CudaIPCSentDataLimbo final {
  ~CudaIPCSentDataLimbo();
  bool collect();
  void add(std::unique_ptr<CudaIPCSentData> shared_block);
  uint64_t size();

 private:
  // TODO: Can be changed to FIFO in order to avoid full traverse on every
  // collect()
  std::vector<std::unique_ptr<CudaIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

using CudaIPCRefCountersFile = at::ipc::RefCountersFile<int64_t, CudaCounterOps>;

} // namespace
} // namespace torch

namespace c10 {
namespace {
class CudaIPCCollectCallback : public FreeMemoryCallback {
 public:
  bool Execute() override {
    return torch::CudaIPCCollect();
  }
};
} // namespace

} // namespace c10

#endif
