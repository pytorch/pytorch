#pragma once

#include <ATen/ATen.h>

#include <c10/core/Allocator.h>
#include <c10/util/ThreadLocalDebugInfo.h>

class TestMemoryReportingInfo : public c10::MemoryReportingInfoBase {
 public:
  struct Record {
    void* ptr;
    int64_t alloc_size;
    size_t total_allocated;
    size_t total_reserved;
    c10::Device device;
  };

  std::vector<Record> records;

  TestMemoryReportingInfo() = default;
  ~TestMemoryReportingInfo() override = default;

  void reportMemoryUsage(
      void* ptr,
      int64_t alloc_size,
      size_t total_allocated,
      size_t total_reserved,
      c10::Device device) override {
    records.emplace_back(
        Record{ptr, alloc_size, total_allocated, total_reserved, device});
  }

  bool memoryProfilingEnabled() const override {
    return true;
  }

  Record getLatestRecord() {
    return records.back();
  }
};
