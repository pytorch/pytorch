#include <ATen/ATen.h>
#include <ATen/native/cuda/stochastic_rounding.cuh>


namespace at {
namespace native {

template <typename input_t, typename output_t>
__global__ void stochastic_rounding_kernel(
    const input_t* input,
    output_t* output,
    const int64_t numel,
    std::pair<uint64_t, uint64_t> seed_and_offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed_and_offset.first, tid, seed_and_offset.second, &state);

  round_stochastically<output_t, input_t, at::Half> rounder;

  for (int64_t i = tid; i < numel; i += blockDim.x * gridDim.x) {
    output[i] = rounder(input[i], curand_uniform(&state));
  }
}

Tensor stochastic_rounding_cuda(const Tensor& input, c10::optional<Generator> gen_) {

  TORCH_CHECK(input.is_contiguous());

  if (input.scalar_type() == kHalf) {
    return input;
  }

  Tensor output = at::empty_like(input, input.options().dtype(kHalf), input.suggest_memory_format());
  const int64_t numel = input.numel();
  if (numel == 0) {
    return output;
  }

  const int block = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block;
  unsigned int grid = (numel + block - 1) / block;
  grid = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs((numel + block * grid - 1) / (block * grid));
  }

  AT_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),  "stochastic_rounding_cuda", [&] {
      stochastic_rounding_kernel<scalar_t, at::Half><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<at::Half>(),
        numel, rng_engine_inputs);
    });

  return output;
}

} // namespace native
} // namespace at
