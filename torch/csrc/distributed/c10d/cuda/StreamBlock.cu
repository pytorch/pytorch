#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/native/TensorFactories.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>
#include <torch/csrc/distributed/c10d/cuda/StreamBlock.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/zeros.h>
#endif

namespace c10d::cuda::detail {

__device__ void nanosleep(int64_t ns) {
  // This is a noop on pre-CUDA-7.0 and ROCm devices and effectively falls back
  // to a spinlock. This only can sleep for a max of 1ms on CUDA devices.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  __nanosleep(ns);
#endif
}

__device__ int32_t load_cpu_int32(int32_t* ptr) {
#if defined(USE_ROCM)
  // WARNING: this may not be safe
  return atomicAdd_system(ptr, 0);
#else
  int32_t current_value = 0;

  // Bypass L1 cache to see updates at L2 and above.
  // This could use .cv to bypass L2 cache but that's significantly more
  // expensive and the CPU write will clear the L2 cache.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators
  asm volatile("ld.cg.s32 %0, [%1];"
               : "=r"(current_value) // Output operand
               : "l"(ptr) // Input operand
  );
  return current_value;
#endif
}

__device__ void store_cpu_int32(int32_t* ptr, int32_t val) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700))
  // WARNING: this value may be cached without .release
  *ptr = val;
#else
  // Releases memory so it can be seen by other threads on the system.
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#release-acquire-patterns
  asm volatile("st.release.sys.s32 [%0], %1;" ::"l"(ptr), "r"(val));
#endif
}

__global__
// set launch bounds to limit to 1 thread per block, 1 block per MP
__launch_bounds__(1, 1) void kernel_barrier(int32_t* value, size_t timeout_ms) {
  store_cpu_int32(&value[1], StreamBlockStatus::RUNNING);

  size_t start = c10d::symmetric_memory::global_timer_ns();
  size_t timeout_ns = timeout_ms * 1e6; // Convert milliseconds to nanoseconds
  while (true) {
    // Atomically read the value
    int32_t current_value = load_cpu_int32(value);
    // Check if the value is equal to the expected value
    if (current_value == 1) {
      store_cpu_int32(&value[1], StreamBlockStatus::ABORTED);
      return;
    }

    if (timeout_ms > 0) {
      // Check if timeout has been reached
      size_t now = c10d::symmetric_memory::global_timer_ns();
      if ((now - start) > timeout_ns) {
        store_cpu_int32(&value[1], StreamBlockStatus::TIMED_OUT);
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
      at::zeros({2}, at::TensorOptions().dtype(at::kInt)).pin_memory()
    },
      timeout_{timeout} {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* ptr = comm_.mutable_data_ptr<int32_t>();
  auto* ctx = comm_.storage().data_ptr().get_context();

  // grid size 1, block size 1, 0 bytes of shared memory
  kernel_barrier<<<1, 1, 0, stream>>>(ptr, timeout_.count());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // This object may be deallocated before the CUDA kernel completes. We need to
  // register the CPU tensor so it's only freed after the kernel completes
  // execution.
  at::getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
}

C10_REGISTER_CLASS(StreamBlockRegistry, CUDA, StreamBlock)

} // namespace c10d::cuda::detail
