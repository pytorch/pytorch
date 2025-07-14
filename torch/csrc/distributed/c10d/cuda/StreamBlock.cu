#include <ATen/native/TensorFactories.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/cuda/StreamBlock.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace c10d::cuda::detail {

__device__ void nanosleep(int64_t ns) {
  // This is a noop on pre-CUDA-7.0 and ROCm devices and effectively falls back
  // to a spinlock. This only can sleep for a max of 1ms on CUDA devices.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  __nanosleep(ns);
#endif
}

__global__
// set launch bounds to limit to 1 thread per block, 1 block per MP
__launch_bounds__(1, 1) void kernel_barrier(int32_t* value, size_t timeout_ms) {
  value[1] = StreamBlockStatus::RUNNING;

  size_t start = c10d::symmetric_memory::global_timer_ns();
  size_t timeout_ns = timeout_ms * 1e6; // Convert milliseconds to nanoseconds
  while (true) {
    // Atomically read the value
    int current_value = atomicAdd(&value[0], 0);
    // Check if the value is equal to the expected value
    if (current_value == 1) {
      value[1] = StreamBlockStatus::ABORTED;
      return;
    }

    if (timeout_ms > 0) {
      // Check if timeout has been reached
      size_t now = c10d::symmetric_memory::global_timer_ns();
      if ((now - start) > timeout_ns) {
        value[1] = StreamBlockStatus::TIMED_OUT;
        return;
      }
    }

    // sleep for 1ms
    nanosleep(1000000);
  }
}

StreamBlock::StreamBlock(std::chrono::milliseconds timeout)
    : comm_{
      // We need to pin the memory since we access the CPU memory directly form
      // the GPU.
      at::empty({2}, at::TensorOptions().dtype(at::kInt)).pin_memory()
    },
      timeout_{timeout} {
  // grid size 1, block size 1, 0 bytes of shared memory
  kernel_barrier<<<1, 1, 0>>>(
      comm_.mutable_data_ptr<int32_t>(), timeout_.count());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

C10_REGISTER_CLASS(StreamBlockRegistry, CUDA, StreamBlock)

} // namespace c10d::cuda::detail
