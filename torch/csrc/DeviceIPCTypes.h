#pragma once
#include <c10/core/Allocator.h>
#include <torch/csrc/Export.h>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace torch {

TORCH_API bool DeviceIPCCollect();

struct DeviceIPCReceivedData final {
  DeviceIPCReceivedData() = default;
  explicit DeviceIPCReceivedData(std::shared_ptr<void> shared_ptr)
      : shared_ptr_(std::move(shared_ptr)) {}
  std::shared_ptr<void> shared_ptr_;
};

struct DeviceIPCSentData final {
  std::string handle_;
  uint64_t offset_;
  uint64_t* counter_ptr_;
  at::DataPtr original_ptr_;
  // Empty string means no event-based sync required.
  std::string event_bytes_;
  at::Device device_;

  DeviceIPCSentData(
      std::string handle,
      uint64_t offset,
      uint64_t* counter_ptr,
      at::Device device);
  ~DeviceIPCSentData();

  uint64_t counter_value();

  const std::string& handle() const {
    return handle_;
  }
  uint64_t offset() const {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
  void set_event_bytes(std::string bytes) {
    event_bytes_ = std::move(bytes);
  }
  const std::string& event_bytes() const {
    return event_bytes_;
  }
};

TORCH_API at::DataPtr GetNewRefCountedSentDataForDevice(
    void* data,
    at::Device device);

namespace {

inline constexpr int64_t DEVICE_IPC_REF_COUNTER_FILE_SIZE = 10000;
inline constexpr int64_t DEVICE_IPC_WARN_AFTER_X_BLOCKS_IN_LIMBO = 1000;

// All to be deleted data blocks with non-zero reference counter go there
struct DeviceIPCSentDataLimbo final {
  ~DeviceIPCSentDataLimbo();
  bool collect();
  void add(std::unique_ptr<DeviceIPCSentData> shared_block);
  uint64_t size();

 private:
  std::vector<std::unique_ptr<DeviceIPCSentData>> shared_blocks_;
  std::mutex limbo_mutex_;
};

struct DeviceIPCRefCountersFile final {
  DeviceIPCRefCountersFile(
      std::string handle,
      uint64_t size,
      at::DataPtr data_ptr)
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

  void return_offset(uint64_t /* offset */) {
    used_slots_--;
  }

  const std::string& handle() const {
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