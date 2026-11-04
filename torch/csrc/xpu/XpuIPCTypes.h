#pragma once
#ifdef USE_XPU
#include <c10/core/Allocator.h>
#include <c10/xpu/XPUEvent.h>
#include <torch/csrc/Export.h>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// XPU counterpart of torch/csrc/CudaIPCTypes.h.
// See Note [CUDA IPC Refcounting implementation explained]
namespace torch {

TORCH_PYTHON_API bool XpuIPCCollect();

struct XpuIPCReceivedData final {
  XpuIPCReceivedData() = default;
  explicit XpuIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct XpuIPCSentData final {
  std::string handle_;
  uint64_t offset_;
  uint64_t* counter_ptr_; // Reference counter shared memory block
  at::DataPtr original_ptr_; // Original mem allocation
  std::optional<c10::xpu::XPUEvent> event_;
  std::string event_handle_; // Serialized IPC handle of event_
  bool event_sync_required_;
  at::Device device_;

  XpuIPCSentData(
      std::string handle,
      uint64_t offset,
      uint64_t* counter_ptr,
      at::Device device);
  ~XpuIPCSentData();

  uint64_t counter_value();
  std::string handle() {
    return handle_;
  }
  uint64_t offset() {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

TORCH_PYTHON_API at::DataPtr GetNewRefCountedSentXpuData(
    void* data,
    at::Device device);

inline constexpr int64_t XPU_IPC_REF_COUNTER_FILE_SIZE = 10000;

namespace {

inline constexpr int64_t XPU_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;
inline constexpr int64_t XPU_IPC_MAXIMUM_EVENTS_TO_USE = 1000;

// All to be deleted data blocks with non zero reference counter goes there
struct XpuIPCSentDataLimbo final {
  ~XpuIPCSentDataLimbo();
  bool collect();
  void add(std::unique_ptr<XpuIPCSentData> shared_block);
  uint64_t size();

 private:
  std::vector<std::unique_ptr<XpuIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct XpuIPCRefCountersFile final {
  XpuIPCRefCountersFile(std::string handle, uint64_t size, at::DataPtr data_ptr)
      : size_(size),
        handle_(std::move(handle)),
        refcounted_shared_mem_(std::move(data_ptr)) {}

  uint64_t* counter_ptr() {
    return static_cast<uint64_t*>(refcounted_shared_mem_.get()) + next_offset_;
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

  uint64_t get_offset() {
    return next_offset_;
  }

  void rotate_offset() {
    next_offset_++;
    used_slots_++;
  }

  void return_offset(uint64_t /*offset*/) {
    used_slots_--;
  }

  std::string handle() {
    return handle_;
  }

 private:
  uint64_t next_offset_{0};
  uint64_t size_;
  uint64_t used_slots_{0};
  std::string handle_;
  at::DataPtr refcounted_shared_mem_;
};

} // namespace
} // namespace torch

#endif
