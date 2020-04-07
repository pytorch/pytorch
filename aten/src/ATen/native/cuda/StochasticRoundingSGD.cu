#include <ATen/ATen.h>
#include <ATen/native/cuda/stochastic_rounding.cuh>


namespace at {
namespace native {

// SGD update math with Stochastic Rounding
template <typename scalar_t>
__global__ void stochastic_rounding_sgd_step_kernel(
    scalar_t *weights, scalar_t *gradients, scalar_t *momentum_buffer,
    float* inv_scale, float* found_inf,
    float weight_decay, float momentum, float dampening, float lr,
    bool nesterov, bool first_run, int numel, std::pair<uint64_t, uint64_t> seeds) {

  if (*found_inf) return;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  for (int i = tid; i < numel; i += blockDim.x * gridDim.x) {
    float weight = static_cast<float>(weights[i]);
    float gradient = static_cast<float>(gradients[i]) * (*inv_scale);
    float velocity = static_cast<float>(momentum_buffer[i]);
    float4 random_values = curand_uniform4(&state);

    if (weight_decay != 0.0f)
      gradient += weight_decay * weight;

    if (momentum != 0.0f) {
      if (!first_run)
        velocity = velocity * momentum + (1.0f - dampening) * gradient;
      else
        velocity = gradient;

      if (nesterov)
        gradient += momentum * velocity;
      else
        gradient = velocity;
    }

    weight -= lr * gradient;

    weights[i] = round_stochastically<scalar_t>(weight, random_values.x);
    if (momentum != 0.0f)
      momentum_buffer[i] = round_stochastically<scalar_t>(velocity, random_values.y);
  }
}

Tensor stochastic_rounding_sgd_step_cuda(
    Tensor& param, const Tensor& grad, Tensor& momentum_buffer,
    const Tensor& inv_scale, const Tensor& found_inf,
    double lr, double momentum, double weight_decay, double dampening,
    bool nesterov, bool first_run, Generator gen_) {

  if (param.numel() == 0) return param;

  TORCH_CHECK(param.is_contiguous());
  TORCH_CHECK(grad.is_contiguous());
  TORCH_CHECK(momentum_buffer.is_contiguous());

  const int64_t numel = param.numel();
  const int block_size = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  uint64_t counter_offset = ((numel + dim_block.x * grid.x - 1) / (dim_block.x * grid.x)) * 4;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        param.scalar_type(), "stochastic_rounding_sgd_step_cuda", [&] {
        stochastic_rounding_sgd_step_kernel<scalar_t><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            momentum_buffer.data_ptr<scalar_t>(),
            inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
            static_cast<float>(weight_decay), static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr),
            nesterov, first_run, numel, rng_engine_inputs);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return param;
}

} // namespace native
} // namespace at
