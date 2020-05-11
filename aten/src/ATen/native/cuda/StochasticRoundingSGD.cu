#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/stochastic_rounding.cuh>


namespace at {
namespace native {

// SGD update math with Stochastic Rounding
template <typename scalar_t, typename IndexType, int ADims>
__global__ void stochastic_rounding_sgd_step_kernel(
    cuda::detail::TensorInfo<scalar_t, IndexType> weights,
    cuda::detail::TensorInfo<scalar_t, IndexType> gradients,
    cuda::detail::TensorInfo<scalar_t, IndexType> momentum_buffer,
    float* inv_scale, float* found_inf,
    float weight_decay, float momentum, float dampening, float lr,
    bool nesterov, bool first_run, int numel, std::pair<uint64_t, uint64_t> seeds) {

  if (*found_inf) return;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  round_stochastically<scalar_t, float, at::Half> rounder;

  for (int i = tid; i < numel; i += blockDim.x * gridDim.x) {
    const IndexType w_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, weights);
    float weight = static_cast<float>(weights.data[w_offset]);
    const IndexType g_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, gradients);
    float gradient = static_cast<float>(gradients.data[g_offset]) * (*inv_scale);
    const IndexType v_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, momentum_buffer);
    float velocity = static_cast<float>(momentum_buffer.data[v_offset]);
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

    weights.data[w_offset] = rounder(weight, random_values.x);
    if (momentum != 0.0f)
      momentum_buffer.data[v_offset] = rounder(velocity, random_values.y);
  }
}

Tensor stochastic_rounding_sgd_step_cuda(
    Tensor& param, const Tensor& grad, Tensor& momentum_buffer,
    const Tensor& inv_scale, const Tensor& found_inf,
    double lr, double momentum, double weight_decay, double dampening,
    bool nesterov, bool first_run, c10::optional<Generator> gen_) {

  if (param.numel() == 0) return param;

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

  if (cuda::detail::canUse32BitIndexMath(param)) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "stochastic_rounding_sgd_step_cuda", [&] {
      auto param_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(param);
      param_info.collapseDims();
      auto grad_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(grad);
      grad_info.collapseDims();
      auto momentum_buffer_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(momentum_buffer);
      momentum_buffer_info.collapseDims();

      switch (param_info.dims) {
        case 1:
          stochastic_rounding_sgd_step_kernel<scalar_t, unsigned int, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
              param_info, grad_info, momentum_buffer_info,
              inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
              weight_decay, momentum, dampening, lr, nesterov, first_run, numel, rng_engine_inputs);
          break;
        default:
          stochastic_rounding_sgd_step_kernel<scalar_t, unsigned int, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
              param_info, grad_info, momentum_buffer_info,
              inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
              weight_decay, momentum, dampening, lr, nesterov, first_run, numel, rng_engine_inputs);
      }
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "stochastic_rounding_sgd_step_cuda", [&] {
        auto param_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(param);
        param_info.collapseDims();
        auto grad_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(grad);
        grad_info.collapseDims();
        auto momentum_buffer_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(momentum_buffer);
        momentum_buffer_info.collapseDims();

        switch (param_info.dims) {
          case 1:
            stochastic_rounding_sgd_step_kernel<scalar_t, uint64_t, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                weight_decay, momentum, dampening, lr, nesterov, first_run, numel, rng_engine_inputs);
            break;
          default:
            stochastic_rounding_sgd_step_kernel<scalar_t, uint64_t, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, momentum_buffer_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                weight_decay, momentum, dampening, lr, nesterov, first_run, numel, rng_engine_inputs);
        }
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return param;
}

} // namespace native
} // namespace at
