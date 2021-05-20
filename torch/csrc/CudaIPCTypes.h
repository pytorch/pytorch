#pragma once
#ifdef USE_CUDA
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Logging.h>
#include <cuda_runtime_api.h>
#include <cstddef>
namespace torch {

TORCH_CUDA_CU_API bool CudaIPCCollect();

struct CudaIPCReceivedData final {
  explicit CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct CudaIPCSentData final {
  std::string handle_;
  int64_t offset_;
  int64_t* counter_ptr_; // Reference counter shared memory block
  at::DataPtr original_ptr_; // Original mem allocation
  cudaEvent_t event_; // Sync cuEventDestroy
  bool event_sync_required_;
  at::Device device_;

  CudaIPCSentData(
      std::string handle,
      int64_t offset,
      int64_t* counter_ptr,
      at::Device device);
  ~CudaIPCSentData();

  int64_t counter_value();
  std::string handle() {
    return handle_;
  }
  int64_t offset() {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

TORCH_CUDA_CU_API at::DataPtr GetNewRefCountedSentData(void* data, at::Device device);

namespace {

constexpr int64_t CUDA_IPC_REF_COUNTER_FILE_SIZE = 10000;
constexpr int64_t CUDA_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;
// This was determined empirically that CUDA (v10.1 and below) have the limit
// on the number of recorded blocking interprocess events. It is around ~22,000.
// And to give us leeway, we picked 1000 as it gives us enough events to share
// tensors effectively.
constexpr int64_t CUDA_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

// All to be deleted data blocks with non zero reference counter goes there
struct CudaIPCSentDataLimbo final {
  ~CudaIPCSentDataLimbo();
  bool collect();
  void clear_shared_blocks();
  void add(std::unique_ptr<CudaIPCSentData> shared_block);
  uint64_t size() {
    return shared_blocks_.size();
  }

 private:
  // TODO: Can be changed to FIFO in order to avoid full traverse on every
  // collect()
  std::vector<std::unique_ptr<CudaIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct CudaIPCRefCountersFile final {
  CudaIPCRefCountersFile(
      // NOLINTNEXTLINE(modernize-pass-by-value)
      std::string handle,
      uint64_t size,
      at::DataPtr data_ptr)
      : next_offset_(0),
        size_(size),
        used_slots_(0),
        handle_(handle),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  int64_t* counter_ptr() {
    return static_cast<int64_t*>(refcounted_shared_mem_.get()) + next_offset_;
  }

  void set_counter(uint64_t value) {
    *counter_ptr() = value;
  }

  bool have_offsets() {
    return next_offset_ < size_;
  }

  bool offsets_in_use() {
    return used_slots_;
  }

  int64_t get_offset() {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t offset /* unused */) {
    used_slots_--;
  }

  std::string handle() {
    return handle_;
  }

 private:
  uint64_t next_offset_;
  uint64_t size_;
  uint64_t used_slots_;
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

} // namespace
} // namespace torch

namespace c10 {
namespace {
class CudaIPCCollectCallback : public FreeMemoryCallback {
 public:
  // NOLINTNEXTLINE(modernize-use-override,modernize-use-equals-default)
  ~CudaIPCCollectCallback() {};
  bool Execute() override {
    return torch::CudaIPCCollect();
  }
};
} // namespace

} // namespace c10

#endif
