#include <ATen/ATen.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/stochastic_rounding.cuh>


namespace at {
namespace native {

template <typename scalar_t, typename IndexType, int ADims>
__global__ void stochastic_rounding_adam_step_kernel(
    cuda::detail::TensorInfo<scalar_t, IndexType> weights,
    const cuda::detail::TensorInfo<scalar_t, IndexType> gradients,
    cuda::detail::TensorInfo<scalar_t, IndexType> exp_avg,
    cuda::detail::TensorInfo<scalar_t, IndexType> exp_avg_sq,
    cuda::detail::TensorInfo<scalar_t, IndexType> max_exp_avg_sq,
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
    const IndexType w_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, weights);
    float weight = static_cast<float>(weights.data[w_offset]);
    const IndexType g_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, gradients);
    float gradient = static_cast<float>(gradients.data[g_offset]) * (*inv_scale);
    const IndexType m_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, exp_avg);
    float m = static_cast<float>(exp_avg.data[m_offset]);
    // Stochastic Rounding Adam tracks square root of the exponential average of squared gradient.
    const IndexType v_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, exp_avg_sq);
    float v = static_cast<float>(exp_avg_sq.data[v_offset]);
    v = v * v;
    IndexType max_v_offset = 0;
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
      max_v_offset = cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(i, max_exp_avg_sq);
      float prev_max_v = static_cast<float>(max_exp_avg_sq.data[max_v_offset]);
      prev_max_v = prev_max_v * prev_max_v;
      max_v = fmaxf(prev_max_v, v);
    }

    weight -= (lr / m_correction) * m / (sqrtf(max_v / v_correction) + eps);

    weights.data[w_offset] = rounder(weight, random_values.x);
    exp_avg.data[m_offset] = rounder(m, random_values.y);
    exp_avg_sq.data[v_offset] = rounder(sqrtf(v), random_values.z);
    if (is_amsgrad) {
      max_exp_avg_sq.data[max_v_offset] = rounder(sqrtf(max_v), random_values.w);
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

  if (cuda::detail::canUse32BitIndexMath(param)) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "stochastic_rounding_adam_step_cuda", [&] {
        auto param_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(param);
        param_info.collapseDims();
        auto grad_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(grad);
        grad_info.collapseDims();
        auto exp_avg_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(exp_avg);
        exp_avg_info.collapseDims();
        auto exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(exp_avg_sq);
        exp_avg_sq_info.collapseDims();
        auto max_exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, unsigned int>(max_exp_avg_sq);
        max_exp_avg_sq_info.collapseDims();

        switch (param_info.dims) {
          case 1:
            stochastic_rounding_adam_step_kernel<scalar_t, unsigned int, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                lr, beta1, beta2, weight_decay, eps, step, is_decoupled, is_amsgrad, numel, rng_engine_inputs);
            break;
          default:
            stochastic_rounding_adam_step_kernel<scalar_t, unsigned int, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                lr, beta1, beta2, weight_decay, eps, step, is_decoupled, is_amsgrad, numel, rng_engine_inputs);
        }
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(param.scalar_type(), "stochastic_rounding_adam_step_cuda", [&] {
        auto param_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(param);
        param_info.collapseDims();
        auto grad_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(grad);
        grad_info.collapseDims();
        auto exp_avg_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(exp_avg);
        exp_avg_info.collapseDims();
        auto exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(exp_avg_sq);
        exp_avg_sq_info.collapseDims();
        auto max_exp_avg_sq_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(max_exp_avg_sq);
        max_exp_avg_sq_info.collapseDims();

        switch (param_info.dims) {
          case 1:
            stochastic_rounding_adam_step_kernel<scalar_t, uint64_t, 1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                lr, beta1, beta2, weight_decay, eps, step, is_decoupled, is_amsgrad, numel, rng_engine_inputs);
            break;
          default:
            stochastic_rounding_adam_step_kernel<scalar_t, uint64_t, -1><<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                param_info, grad_info, exp_avg_info, exp_avg_sq_info, max_exp_avg_sq_info,
                inv_scale.data_ptr<float>(), found_inf.data_ptr<float>(),
                lr, beta1, beta2, weight_decay, eps, step, is_decoupled, is_amsgrad, numel, rng_engine_inputs);
        }
    });
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return param;
}

} // namespace native
} // namespace at
