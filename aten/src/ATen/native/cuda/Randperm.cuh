#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/Utils.h>

#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace {

// See note [Algorithm of randperm]
template<typename T, typename scalar_t>
__global__ void randperm_handle_duplicate_keys_kernel(T *keys, scalar_t *data, T mask, int n, at::PhiloxCudaState philox_args) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  // find the beginning of islands
  if (tid >= n - 1) return;  // out of range
  if ((keys[tid] & mask) != (keys[tid + 1] & mask)) return;  // not in an island
  if (tid != 0 && (keys[tid] & mask) == (keys[tid - 1] & mask)) return;  // not the beginning of an island

  // find the size of islands
  int island_size = 0;
  do { island_size++; }
  while ((tid + island_size < n) && (keys[tid + island_size] & mask) == (keys[tid] & mask));

  // do random permutation inside each island.
  data += tid;
  auto seeds = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds), tid, std::get<1>(seeds), &state);
  for (int i = island_size - 1; i > 0; i--) {
    unsigned int r = curand(&state) % (i + 1);
    if (i != r) {
      scalar_t tmp = data[i];
      data[i] = data[r];
      data[r] = tmp;
    }
  }
}

// See note [Algorithm of randperm]
template<typename T, typename scalar_t>
void randperm_handle_duplicate_keys(T *keys, scalar_t *data, int bits, int64_t n, std::optional<at::Generator> &gen_) {
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());
  int64_t counter_offset = n;
  at::PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }
  T mask = static_cast<T>((1UL << bits) - 1);
  randperm_handle_duplicate_keys_kernel<<<(n + 511) / 512, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
    keys, data, mask, n, rng_engine_inputs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
