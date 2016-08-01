#include "caffe2/core/context_gpu.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/init.h"

namespace caffe2 {
thread_local ThreadLocalCUDAObjects CUDAContext::cuda_objects_;

namespace {
bool Caffe2UsePinnedCPUAllocator(int*, char***) {
#ifdef __SANITIZE_ADDRESS__
  // Note(jiayq): for more details, see
  //     https://github.com/google/sanitizers/issues/629
  LOG(WARNING) << "There are known issues between address sanitizer and "
                  "cudaMallocHost. As a result, caffe2 will not enable pinned "
                  "memory allocation in asan mode. If you are expecting any "
                  "behavior that depends on asan, be advised that it is not "
                  "turned on.";
  return true;
#else
  if (!HasCudaGPU()) {
    VLOG(1) << "No GPU present. I won't use pinned allocator then.";
    return true;
  }
  VLOG(1) << "Caffe2 gpu: setting CPUAllocator to PinnedCPUAllocator.";
  SetCPUAllocator(new PinnedCPUAllocator());
  return true;
#endif
}

REGISTER_CAFFE2_INIT_FUNCTION(Caffe2UsePinnedCPUAllocator,
                              &Caffe2UsePinnedCPUAllocator,
                              "Make the CPU side use pinned memory.");
}  // namespace
}  // namespace caffe2
