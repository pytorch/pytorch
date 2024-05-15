#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/rrelu_with_noise_native.h>
#endif


namespace at::native {

template <typename scalar_t, int unroll_factor, typename F>
#if __CUDA_ARCH__ >= 350 || defined USE_ROCM
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void rrelu_with_noise_cuda_kernel(
    int numel,
    PhiloxCudaState philox_args,
    scalar_t* output,
    const scalar_t* input,
    scalar_t* noise,
    double lower,
    double upper,
    const F& random_func) {
  auto seeds = at::cuda::philox::unpack(philox_args);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds),
              idx,
              std::get<1>(seeds),
              &state);

  int grid_stride = blockDim.x * gridDim.x * unroll_factor;
  int rounded_size = ((numel - 1) / grid_stride + 1) * grid_stride;
  double range = upper - lower;

  for (int linear_index = idx; linear_index < rounded_size; linear_index += grid_stride) {
    auto rand = random_func(&state);

    // ensure that (&rand.x)[ii] is safe
    static_assert(sizeof(rand)/sizeof(rand.x) == unroll_factor, "");

    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li >= numel) {
        continue;
      }
      scalar_t r = static_cast<scalar_t>((&rand.x)[ii]);
      r = r * range + lower;
      if (input[li] <= 0) {
        output[li] = input[li] * r;
        noise[li] = r;
      } else {
        output[li] = input[li];
        noise[li] = static_cast<scalar_t>(1);
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t>
inline void _rrelu_with_noise_cuda_train(
    Tensor& output,
    const Tensor& input_,
    const Tensor& noise_,
    const Scalar& lower_,
    const Scalar& upper_,
    std::optional<Generator> generator) {
  auto input = input_.contiguous();
  auto noise = noise_.contiguous();
  Tensor tmp_output = output.contiguous();

  int64_t numel = input.numel();
  auto execution_policy = calc_execution_policy(numel);

  auto counter_offset = std::get<0>(execution_policy);
  auto grid = std::get<1>(execution_policy);
  auto block = std::get<2>(execution_policy);

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(
      generator, cuda::detail::getDefaultCUDAGenerator());
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  scalar_t* noise_data = noise.mutable_data_ptr<scalar_t>();
  scalar_t* output_data = tmp_output.mutable_data_ptr<scalar_t>();

  double lower = lower_.to<double>();
  double upper = upper_.to<double>();

  auto stream = at::cuda::getCurrentCUDAStream();

  if (std::is_same<scalar_t, double>::value) {
    rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
        numel,
        rng_engine_inputs,
        output_data,
        input_data,
        noise_data,
        lower,
        upper,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand_uniform2_double(state);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // half and float
    rrelu_with_noise_cuda_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
        numel,
        rng_engine_inputs,
        output_data,
        input_data,
        noise_data,
        lower, upper,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
          return curand_uniform4(state);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (!output.is_contiguous()) {
    output.copy_(tmp_output);
  }
}

Tensor& rrelu_with_noise_out_cuda(const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator,
    Tensor& output) {
  at::native::resize_output(output, self.sizes());

  if (self.numel() == 0) {
    return output;
  }

  TensorArg self_arg{self, "self", 1}, noise_arg{noise, "noise", 2},
      output_arg{output, "output", 3};
  checkAllSameGPU("rrelu_with_noise_out_cuda", {self_arg, noise_arg, output_arg});

  if (training) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        self.scalar_type(), "rrelu_with_noise_out_cuda", [&] {
          _rrelu_with_noise_cuda_train<scalar_t>(
              output, self, noise, lower, upper, generator);
        });
  }
  else {
    auto lower_tensor = lower.to<double>();
    auto upper_tensor = upper.to<double>();
    Scalar negative_slope = (lower_tensor + upper_tensor) / 2;
    at::leaky_relu_out(output, self, negative_slope);
  }
  return output;
}

Tensor rrelu_with_noise_cuda(
    const Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  Tensor output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::native::rrelu_with_noise_out_cuda(self, noise, lower, upper, training, generator, output);
}

Tensor& rrelu_with_noise_cuda_(
    Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  return at::native::rrelu_with_noise_out_cuda(
      self, noise, lower, upper, training, generator, self);
}

}  // namespace at::native
