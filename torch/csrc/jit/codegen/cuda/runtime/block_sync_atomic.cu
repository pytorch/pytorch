
// Counter-based block synchronization. Only meant to be used for
// debugging and validating synchronization. This should be replaced
// with cuda::barrier::arrive_and_wait as that should be more robust.

namespace block_sync {

using CounterType = unsigned int;
static constexpr CounterType COUNTER_TYPE_MAX = ~(CounterType)0;
__shared__ CounterType sync_counter;

__device__ void init() {
  const unsigned int tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
  if (tid == 0) {
    sync_counter = 0;
  }
  __syncthreads();
}

// Emulate __syncthreads() with a synchronization counter
__device__ void sync() {
  unsigned int backoff = 8;
  const unsigned int backoff_max = 256;
  const unsigned int num_threads = blockDim.x * blockDim.y * blockDim.z;

  __threadfence_block();

  // Use counter range only up to a limit so that the next val won't
  // overflow.

  const auto counter_max = (COUNTER_TYPE_MAX / num_threads) * num_threads;
  const auto old = atomicInc(&sync_counter, counter_max - 1);

  const auto next = (old / num_threads) * num_threads + num_threads;

  auto local_sync_counter = *(volatile CounterType*)(&sync_counter);

  // sync_counter may wrap around, which means local_sync_counter
  // becomes smaller than old. In that case, it's guaranteed that all
  // threads have incremented the counter.
  while (local_sync_counter < next && old < local_sync_counter) {
#if __CUDA_ARCH__ >= 700
    // __nanosleep only available on compute capability 7.0 or higher
    __nanosleep(backoff); // avoids busy waiting
#endif
    if (backoff < backoff_max) {
      backoff *= 2;
    }
    local_sync_counter = *(volatile CounterType*)(&sync_counter);
  }
}

} // namespace block_sync
