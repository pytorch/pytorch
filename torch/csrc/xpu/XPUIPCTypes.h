#pragma once
#ifdef USE_XPU
#include <c10/core/Allocator.h>
#include <cstdint>
#include <string>

namespace torch {

inline constexpr int64_t XPU_IPC_REF_COUNTER_FILE_SIZE = 10000;

bool XpuIPCCollect();

struct XpuIPCSentData final {
  std::string handle_;
  uint64_t offset_;
  uint64_t* counter_ptr_;
  at::DataPtr original_ptr_;

  XpuIPCSentData(
      std::string handle,
      uint64_t offset,
      uint64_t* counter_ptr,
      at::Device device);
  ~XpuIPCSentData();

  uint64_t counter_value() {
    return *counter_ptr_;
  }
  const std::string& handle() const {
    return handle_;
  }
  uint64_t offset() const {
    return offset_;
  }
  void set_original_ptr(at::DataPtr data_ptr) {
    original_ptr_ = std::move(data_ptr);
  }
};

at::DataPtr GetNewRefCountedSentXpuData(void* data, at::Device device);

} // namespace torch
#endif
