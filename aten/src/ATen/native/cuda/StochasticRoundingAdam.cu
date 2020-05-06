#include <ATen/ATen.h>
#include <ATen/native/cuda/stochastic_rounding.cuh>


namespace at {
namespace native {

template <typename scalar_t>
__global__ void stochastic_rounding_adam_step_kernel(
    scalar_t *weights, scalar_t *gradients,
    scalar_t *exp_avg, scalar_t *exp_avg_sq, scalar_t *max_exp_avg_sq,
    float *inv_scale, float *found_inf,
    float lr, float beta1, float beta2,
    float weight_decay, float eps, int step,
    bool is_decoupled, bool is_amsgrad,
    int numel, std::pair<uint64_t, uint64_t> seeds) {

  if (*found_inf) return;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, tid, seeds.second, &state);

  round_stochastically<scalar_t, float, at::Half> rounder;

  float m_correction = 1.0 - powf(beta1, step);
  float v_correction = 1.0 - powf(beta2, step);

  for  (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
    float weight = static_cast<float>(weights[i]);
    float gradient = static_cast<float>(gradients[i]) * (*inv_scale);
    float m = static_cast<float>(exp_avg[i]);
    // Stochastic Rounding Adam tracks square root of the exponential average of squared gradient.
    float v = static_cast<float>(exp_avg_sq[i]);
    v = v * v;
    float4 random_values = curand_uniform4(&state);

    if (weight_decay != 0.0f) {
      if (is_decoupled)
        weight *= (1 - lr * weight_decay);
      else
        gradient += weight_decay * weight;
    }

    // Update m and v.
    m = beta1 * m + (1.0 - beta1) * gradient;
    v = beta2 * v + (1.0 - beta2) * (gradient * gradient);

    // Unbias v
    float max_v = v;
    if (is_amsgrad) {
      float prev_max_v = static_cast<float>(max_exp_avg_sq[i]);
      prev_max_v = prev_max_v * prev_max_v;
      max_v = fmaxf(prev_max_v, v);
    }

    weight -= (lr / m_correction) * m / (sqrtf(max_v / v_correction) + eps);

    weights[i] = rounder(weight, random_values.x);
    exp_avg[i] = rounder(m, random_values.y);
    exp_avg_sq[i] = rounder(sqrtf(v), random_values.z);
    if (is_amsgrad) {
      max_exp_avg_sq[i] = rounder(sqrtf(max_v), random_values.w);
    }
  }
}


Tensor stochastic_rounding_adam_step_cuda(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    Tensor& max_exp_avg_sq,
    const Tensor& inv_scale,
    const Tensor& found_inf,
    double lr, double beta1, double beta2,
    double weight_decay, double eps, int64_t step,
    bool is_decoupled, bool is_amsgrad, c10::optional<Generator> gen_) {

  if (param.numel() == 0) return param;

  TORCH_CHECK(param.is_contiguous());
  TORCH_CHECK(grad.is_contiguous());
  TORCH_CHECK(exp_avg.is_contiguous());
  TORCH_CHECK(exp_avg_sq.is_contiguous());
  TORCH_CHECK(max_exp_avg_sq.is_contiguous());

  const int64_t numel = param.numel();
  const int block_size = 256;
  const int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());

  uint64_t counter_offset = ((numel + dim_block.x * grid.x - 1) / (block_size * grid.x)) * 4;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      param.scalar_type(), "stochastic_rounding_adam_step_cuda", [&] {
        stochastic_rounding_adam_step_kernel<scalar_t><<<grid, dim_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            param.data_ptr<scalar_t>(),
            grad.data_ptr<scalar_t>(),
            exp_avg.data_ptr<scalar_t>(),
            exp_avg_sq.data_ptr<scalar_t>(),
            max_exp_avg_sq.data_ptr<scalar_t>(),
            inv_scale.data_ptr<float>(),
            found_inf.data_ptr<float>(),
            lr, beta1, beta2, weight_decay, eps, step,
            is_decoupled, is_amsgrad,
            numel, rng_engine_inputs);
        }
      );
  AT_CUDA_CHECK(cudaGetLastError());
  return param;
}

} // namespace native
} // namespace at
