#include <ATen/core/Allocator.h>
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"

C10_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage");

C10_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    true,
    "If set, do memory zerofilling when allocating on CPU");

C10_DEFINE_bool(
    caffe2_cpu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on CPU");

namespace caffe2 {

void memset_junk(void* data, size_t num) {
  // This garbage pattern is NaN when interpretted as floating point values,
  // or as very large integer values.
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
  int32_t int64_count = num / sizeof(kJunkPattern64);
  int32_t remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  for (int i = 0; i < int64_count; i++) {
    data_i64[i] = kJunkPattern64;
  }
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

void NoDelete(void*) {}

at::Allocator* GetCPUAllocator() {
  return GetAllocator(CPU);
}

void SetCPUAllocator(at::Allocator* alloc) {
  SetAllocator(CPU, alloc);
}

// Global default CPU Allocator
static DefaultCPUAllocator g_cpu_alloc;

REGISTER_ALLOCATOR(CPU, &g_cpu_alloc);

MemoryAllocationReporter DefaultCPUAllocator::reporter_;

void MemoryAllocationReporter::New(void* ptr, size_t nbytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  size_table_[ptr] = nbytes;
  allocated_ += nbytes;
  LOG(INFO) << "Caffe2 alloc " << nbytes << " bytes, total alloc " << allocated_
            << " bytes.";
}

void MemoryAllocationReporter::Delete(void* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = size_table_.find(ptr);
  CHECK(it != size_table_.end());
  allocated_ -= it->second;
  LOG(INFO) << "Caffe2 deleted " << it->second << " bytes, total alloc "
            << allocated_ << " bytes.";
  size_table_.erase(it);
}

} // namespace caffe2
