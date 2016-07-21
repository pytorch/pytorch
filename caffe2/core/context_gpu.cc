#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/init.h"

namespace caffe2 {
thread_local ThreadLocalCUDAObjects CUDAContext::cuda_objects_;

namespace {
bool Caffe2UsePinnedCPUAllocator(int*, char***) {
  if (!HasCudaGPU()) {
    VLOG(1) << "No GPU present. I won't use pinned allocator then.";
    return true;
  }
  VLOG(1) << "Caffe2 gpu: setting CPUAllocator to PinnedCPUAllocator.";
  SetCPUAllocator(new PinnedCPUAllocator());
  return true;
}

REGISTER_CAFFE2_INIT_FUNCTION(Caffe2UsePinnedCPUAllocator,
                              &Caffe2UsePinnedCPUAllocator,
                              "Make the CPU side use pinned memory.");
}  // namespace
}  // namespace caffe2
