#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/cuda/Baton.cuh>

namespace c10d::cuda::detail {

__global__
// set launch bounds to limit to 1 thread per block, 1 block per MP, 1 block per
// cluster
__launch_bounds__(1, 1, 1) void kernel_barrier(
    int32_t* value,
    size_t timeout_ms) {
  value[1] = BatonStatus::RUNNING;

  size_t start = c10d::symmetric_memory::global_timer_ns();
  size_t timeout_ns = timeout_ms * 1e6; // Convert milliseconds to nanoseconds
  while (true) {
    // Atomically read the value
    int current_value = atomicAdd(&value[0], 0);
    // Check if the value is equal to the expected value
    if (current_value == 1) {
      value[1] = BatonStatus::ABORTED;
      return;
    }

    if (timeout_ms > 0) {
      // Check if timeout has been reached
      size_t now = c10d::symmetric_memory::global_timer_ns();
      if ((now - start) > timeout_ns) {
        value[1] = BatonStatus::TIMED_OUT;
        return;
      }
    }

    // sleep for 1ms
    __nanosleep(1000000);
  }
}

Baton::Baton(std::chrono::milliseconds timeout)
    : comm_{at::empty({2}, at::TensorOptions().dtype(at::kInt)).pin_memory()},
      timeout_{timeout} {
  // grid size 1, block size 1, 0 bytes of shared memory
  kernel_barrier<<<1, 1, 0>>>(
      comm_.mutable_data_ptr<int32_t>(), timeout_.count());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

C10_REGISTER_CLASS(BatonRegistry, CUDA, Baton)

} // namespace c10d::cuda::detail
