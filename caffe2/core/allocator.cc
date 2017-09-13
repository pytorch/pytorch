#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/typeid.h"

CAFFE2_DEFINE_bool(
    caffe2_report_cpu_memory_usage,
    false,
    "If set, print out detailed memory usage");

CAFFE2_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    true,
    "If set, do memory zerofilling when allocating on CPU");

namespace caffe2 {

void NoDelete(void*) {}

static std::unique_ptr<CPUAllocator> g_cpu_allocator(new DefaultCPUAllocator());
CPUAllocator* GetCPUAllocator() {
  return g_cpu_allocator.get();
}

void SetCPUAllocator(CPUAllocator* alloc) {
  g_cpu_allocator.reset(alloc);
}

MemoryAllocationReporter CPUContext::reporter_;

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
