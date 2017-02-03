#include "caffe2/core/context.h"
#include "caffe2/core/typeid.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

static std::unique_ptr<CPUAllocator> g_cpu_allocator(new DefaultCPUAllocator());
CPUAllocator* GetCPUAllocator() {
  return g_cpu_allocator.get();
}

void SetCPUAllocator(CPUAllocator* alloc) {
  g_cpu_allocator.reset(alloc);
}

}  // namespace caffe2
