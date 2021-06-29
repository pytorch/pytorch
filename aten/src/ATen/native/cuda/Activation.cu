#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/DistributionTemplates.h>
#include <c10/cuda/CUDAMathCompat.h>

namespace at {
namespace native {

// -----------------------------------
// prelu forward
// -----------------------------------
template <typename scalar_t>
void prelu_cuda_kernel_share_weights(
  const Tensor& input,
  Tensor& result,
  const scalar_t* weight_data)
{
  auto iter = TensorIterator::unary_op(result, input);

  at::native::gpu_kernel(iter,
    [weight_data] GPU_LAMBDA (scalar_t input_val) {
        return (input_val > 0) ? input_val : *weight_data * input_val;
    });
}

template <typename scalar_t>
__global__ void prelu_cuda_kernel_multi_weights(
  scalar_t* result_data,
  const scalar_t* input_data,
  const scalar_t* weight_data,
  int64_t input_stride0,
  int64_t input_stride1,
  int64_t input_numel) {

  int64_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearId >= input_numel) return;

  // multiply values at each channel with weight[channel_index]
  int64_t channel = (linearId % input_stride0) / input_stride1;
  scalar_t input_data_val = input_data[linearId];
  result_data[linearId] = (input_data_val > 0) ? input_data_val : weight_data[channel] * input_data_val;
}

Tensor prelu_cuda(const Tensor& self, const Tensor& weight_) {
  TORCH_CHECK(self.is_cuda());
  TORCH_CHECK(weight_.is_cuda());

  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  Tensor result = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto strides = input.strides();

  // case1: shared weight for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "prelu_cuda", [&] {
      prelu_cuda_kernel_share_weights<scalar_t>(
        input,
        result,
        weight.data_ptr<scalar_t>());
    });
  }
  else { // case2: multiple weights, one for each channel
    int64_t input_ndim = input.dim();
    TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    // config to run cuda kernel
    int64_t input_numel = input.numel();
    const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), input_numel));
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    TORCH_CHECK(cuda::getApplyGrid(input_numel, grid, curDevice), "prelu: input too large or too many dimensions");

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "prelu_cuda", [&] {
      prelu_cuda_kernel_multi_weights<scalar_t>
      <<<grid, block, 0, stream>>>(
        result.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        input_stride0,
        input_stride1,
        input_numel);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
  return result;
}

// -----------------------------------
// prelu backward
// -----------------------------------
template <typename scalar_t>
void prelu_cuda_backward_kernel_share_weights(
  const Tensor& input,
  const Tensor& grad_out,
  Tensor& input_grad,
  Tensor& weight_grad_collector,
  const scalar_t* weight_data) {
  at::TensorIterator iter = TensorIteratorConfig()
      .add_output(input_grad)
      .add_output(weight_grad_collector)
      .add_input(input)
      .add_input(grad_out)
      .build();

  // N.B. `std::tuple` does not support `::operator=` on device code.
  gpu_kernel_multiple_outputs(iter, [=] GPU_LAMBDA (scalar_t input, scalar_t grad_out) -> thrust::tuple<scalar_t, scalar_t> {
    scalar_t input_grad = input > 0 ? grad_out : (*weight_data) * grad_out;
    scalar_t weight_grad_collector = input > 0 ? scalar_t(0) : input * grad_out;
    return {input_grad, weight_grad_collector};
  });
}

template <typename scalar_t>
__global__ void prelu_cuda_backward_kernel_multi_weights(
  const scalar_t* input_data,
  const scalar_t* weight_data,
  const scalar_t* grad_out_data,
  scalar_t* input_grad_data,
  scalar_t* weight_grad_collector,
  int64_t input_stride0,
  int64_t input_stride1,
  int64_t input_numel) {

  int64_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearId >= input_numel) return;
  int64_t channel = (linearId % input_stride0) / input_stride1;
  scalar_t input_data_val = input_data[linearId];
  scalar_t grad_out_data_val = grad_out_data[linearId];
  input_grad_data[linearId] = (input_data_val > 0) ? grad_out_data_val : weight_data[channel] * grad_out_data_val;
  weight_grad_collector[linearId] = (input_data_val > 0) ? scalar_t(0) : input_data_val * grad_out_data_val;
}

std::tuple<Tensor, Tensor> prelu_backward_cuda(const Tensor& grad_out_, const Tensor& self, const Tensor& weight_) {
  TORCH_CHECK(grad_out_.is_cuda());
  TORCH_CHECK(self.is_cuda());
  TORCH_CHECK(weight_.is_cuda());

  auto input = self.contiguous();
  auto grad_out = grad_out_.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(grad_out.is_contiguous());

  int64_t weight_num = weight.numel();
  auto strides = input.strides();
  auto dims = input.dim();
  Tensor input_grad = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor weight_grad = at::empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor weight_grad_collector = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // case1: shared parameter for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "prelu_backward_cuda", [&] {
      prelu_cuda_backward_kernel_share_weights<scalar_t>(
        input,
        grad_out,
        input_grad,
        weight_grad_collector,
        weight.data_ptr<scalar_t>());
    });
    weight_grad.fill_(weight_grad_collector.sum());
  }
  else { // case2: multiple parameters, one for each channel
    int64_t input_ndim = input.dim();
    TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    // config to run cuda kernel
    int64_t input_numel = input.numel();
    const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), input_numel));
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    TORCH_CHECK(cuda::getApplyGrid(input_numel, grid, curDevice), "prelu_backward_cuda: input too large or too many dimensions");

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, input.scalar_type(), "prelu_backward_cuda", [&] {
      prelu_cuda_backward_kernel_multi_weights<scalar_t>
      <<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        grad_out.data_ptr<scalar_t>(),
        input_grad.data_ptr<scalar_t>(),
        weight_grad_collector.data_ptr<scalar_t>(),
        input_stride0,
        input_stride1,
        input_numel);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    // update weight_grad
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    if (dims > 2) {
      for(int64_t i = 2; i < dims; i++) reduce_dims.push_back(i);
    }
    weight_grad = weight_grad_collector.sum(reduce_dims);
  }
  return std::tuple<Tensor, Tensor>{input_grad, weight_grad};
}

// -----------------------------------
// rrelu
// -----------------------------------
template <typename scalar_t, int unroll_factor, typename F>
#if __CUDA_ARCH__ >= 350 || defined __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void rrelu_with_noise_cuda_kernel(
    int numel,
    PhiloxCudaState philox_args,
    scalar_t* output,
    scalar_t* input,
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
        noise[li] = static_cast<scalar_t>(0);
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
    c10::optional<Generator> generator) {
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

  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* noise_data = noise.data_ptr<scalar_t>();
  scalar_t* output_data = tmp_output.data_ptr<scalar_t>();

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
    c10::optional<Generator> generator,
    Tensor& output) {
  TensorArg self_arg{self, "self", 1}, noise_arg{noise, "noise", 2},
      output_arg{output, "output", 3};
  checkAllSameGPU("rrelu_with_noise_out_cuda", {self_arg, noise_arg, output_arg});

  if (training) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
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
    c10::optional<Generator> generator) {
  Tensor output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::native::rrelu_with_noise_out_cuda(self, noise, lower, upper, training, generator, output);
}

Tensor& rrelu_with_noise_cuda_(
    Tensor& self,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    c10::optional<Generator> generator) {
  return at::native::rrelu_with_noise_out_cuda(
      self, noise, lower, upper, training, generator, self);
}

// -----------------------------------
// hardshrink
// -----------------------------------
void hardshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardshrink_cuda", [&]() {
    auto lambd = value.to<scalar_t>();
    gpu_kernel(iter, [lambd]GPU_LAMBDA(scalar_t a) -> scalar_t {
      return (a >= -lambd && a <= lambd) ? scalar_t(0) : a;
    });
  });
}

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "softshrink_cuda", [&]() {
    auto lambd = value.to<scalar_t>();
    gpu_kernel(iter, [lambd]GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a > lambd ? a - lambd : (a < -lambd ? a + lambd : scalar_t(0));
    });
  });
}

void shrink_backward_kernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "shrink_backward_cuda", [&]() {
    auto lambd = value.to<scalar_t>();
    gpu_kernel(iter, [lambd]GPU_LAMBDA(scalar_t grad_val, scalar_t self_val) -> scalar_t {
      return (self_val >= -lambd && self_val <= lambd) ? scalar_t(0) : grad_val;
    });
  });
}

void hardtanh_backward_kernel(TensorIterator& iter, const Scalar& min, const Scalar& max) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, iter.dtype(), "hardtanh_backward_cuda", [&]() {
    auto min_val = min.to<scalar_t>();
    auto max_val = max.to<scalar_t>();
    gpu_kernel(iter, [min_val, max_val]GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return (b <= min_val) || (b >= max_val) ? scalar_t(0) : a;
    });
  });
}

void softplus_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "softplus_cuda", [&]() {
    auto beta = beta_.to<scalar_t>();
    auto threshold = threshold_.to<scalar_t>();
    gpu_kernel(iter, [beta, threshold]GPU_LAMBDA(scalar_t a) -> scalar_t {
      return (a * beta) > threshold ? a : static_cast<scalar_t>(::log1p(std::exp(a * beta))) / beta;
    });
  });
}

void softplus_backward_kernel(TensorIteratorBase& iter, const Scalar& beta_, const Scalar& threshold_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "softplus_backward_cuda", [&]() {
    auto beta = beta_.to<scalar_t>();
    auto threshold = threshold_.to<scalar_t>();
    gpu_kernel(iter, [beta, threshold]GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      scalar_t z = std::exp(b * beta);
      return (b * beta) > threshold ? a : a * z / (z + scalar_t(1.));
    });
  });
}

template <typename scalar_t>
void threshold_kernel_impl(TensorIteratorBase& iter, scalar_t threshold, scalar_t value) {
  gpu_kernel_with_scalars(iter, [=]GPU_LAMBDA(scalar_t x, scalar_t other) -> scalar_t {
    return x <= threshold ? value : other;
  });
}

static void threshold_kernel_cuda(TensorIteratorBase& iter, const Scalar& threshold, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "threshold_cuda", [&] {
    threshold_kernel_impl<scalar_t>(iter, threshold.to<scalar_t>(), value.to<scalar_t>());
  });
}

void elu_kernel(TensorIteratorBase& iter, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "elu_cuda", [&]() {
    auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
    auto poscoef = scale.to<scalar_t>();
    auto negiptcoef = input_scale.to<scalar_t>();
    gpu_kernel(iter, [negcoef, poscoef, negiptcoef]GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a > scalar_t(0) ? a * poscoef : (static_cast<scalar_t>(std::exp(a * negiptcoef)) - scalar_t(1.)) * negcoef;
    });
  });
}

void elu_backward_kernel(TensorIteratorBase& iter, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, bool is_result) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "elu_backward_cuda", [&]() {
    auto negcoef = alpha.to<scalar_t>() * scale.to<scalar_t>();
    auto poscoef = scale.to<scalar_t>();
    auto negiptcoef = input_scale.to<scalar_t>();
    gpu_kernel(iter, [negcoef, poscoef, negiptcoef, is_result]GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      if (is_result) {
        return b <= scalar_t(0) ? a * negiptcoef * (b + negcoef) : a * poscoef;
      } else {
        return b <= scalar_t(0) ? a * negiptcoef * negcoef * (static_cast<scalar_t>(std::exp(b * negiptcoef))) : a * poscoef;
      }
    });
  });
}

namespace {

void GeluCUDAKernelImpl(TensorIteratorBase& it) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, it.dtype(), "GeluCUDAKernelImpl", [&]() {
    using T_ACC = acc_type<scalar_t, true>;
    gpu_kernel(it, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
      return static_cast<T_ACC>(x) *
          c10::cuda::compat::normcdf(static_cast<T_ACC>(x));
    });
  });
}

void GeluBackwardCUDAKernelImpl(TensorIteratorBase& it) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      it.dtype(), "GeluBackwardCUDAKernelImpl", [&]() {
        using T_ACC = acc_type<scalar_t, true>;
        gpu_kernel(it, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
          constexpr T_ACC kBeta = M_2_SQRTPI * M_SQRT1_2 * T_ACC(0.5);
          const T_ACC cdf = c10::cuda::compat::normcdf(static_cast<T_ACC>(x));
          const T_ACC pdf =
              c10::cuda::compat::exp(
                  T_ACC(-0.5) * static_cast<T_ACC>(x) * static_cast<T_ACC>(x)) *
              kBeta;
          return static_cast<T_ACC>(dy) * (cdf + static_cast<T_ACC>(x) * pdf);
        });
      });
}

void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "leaky_relu_cuda", [&]() {
    auto negval = negval_.to<scalar_t>();
    gpu_kernel(iter, [negval]GPU_LAMBDA(scalar_t a) -> scalar_t {
      return a > scalar_t(0) ? a : a * negval;
    });
  });
}

void leaky_relu_backward_kernel(TensorIteratorBase& iter, const Scalar& negval_) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "leaky_relu_backward_cuda", [&]() {
    auto negval = negval_.to<scalar_t>();
    gpu_kernel(iter, [negval]GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a > scalar_t(0) ? b : b * negval;
    });
  });
}

void hardswish_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardswish_cuda", [&]() {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC zero(0.0f);
    const T_ACC one_sixth(1.0f / 6.0f);
    const T_ACC three(3.0f);
    const T_ACC six(6.0f);
    gpu_kernel(iter, [zero, one_sixth, three, six]GPU_LAMBDA(scalar_t self_val) -> scalar_t {
      T_ACC x = static_cast<T_ACC>(self_val);
      return x * std::min(std::max(x + three, zero), six) * one_sixth;
    });
  });
}

void hardswish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardswish_backward_cuda", [&]() {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC zero(0.0f);
    const T_ACC three(3.0f);
    const T_ACC neg_three(-3.0f);
    const T_ACC one_half(0.5f);
    gpu_kernel(
      iter,
      [zero, three, neg_three, one_half]GPU_LAMBDA(scalar_t grad_val_, scalar_t self_val_) -> scalar_t {
        T_ACC grad_val = static_cast<T_ACC>(grad_val_);
        T_ACC self_val = static_cast<T_ACC>(self_val_);
        if (self_val < neg_three) {
          return zero;
        } else if (self_val <= three) {
          return grad_val * ((self_val / three) + one_half);
        } else {
          return grad_val;
        }
    });
  });
}

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardsigmoid_cuda", [&]() {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC zero(0.0f);
    const T_ACC one_sixth(1.0f / 6.0f);
    const T_ACC three(3.0f);
    const T_ACC six(6.0f);
    gpu_kernel(iter, [zero, one_sixth, three, six]GPU_LAMBDA(scalar_t self_val) -> scalar_t {
      T_ACC x = static_cast<T_ACC>(self_val);
      return std::min(std::max(x + three, zero), six) * one_sixth;
    });
  });
}

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "hardsigmoid_backward_cuda", [&]() {
    using T_ACC = acc_type<scalar_t, true>;
    const T_ACC zero(0.0f);
    const T_ACC three(3.0f);
    const T_ACC neg_three(-3.0f);
    const T_ACC one_sixth(1.0f / 6.0f);
    gpu_kernel(
      iter,
      [zero, three, neg_three, one_sixth]GPU_LAMBDA(scalar_t grad_val_, scalar_t self_val_) -> scalar_t {
        T_ACC grad_val = static_cast<T_ACC>(grad_val_);
        T_ACC self_val = static_cast<T_ACC>(self_val_);
        return (self_val > neg_three && self_val < three)
          ? grad_val * one_sixth
          : zero;
    });
  });
}

void silu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_cuda",
      [&]() {
        gpu_kernel(
            iter,
            [] GPU_LAMBDA(scalar_t x) -> scalar_t {
              using T_ACC = acc_type<scalar_t, true>;
              const T_ACC x_acc = static_cast<T_ACC>(x);
              return x_acc / (T_ACC(1) + c10::cuda::compat::exp(-x_acc));
            });
      });
}

void silu_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "silu_backward_cuda",
      [&]() {
        gpu_kernel(
            iter,
            [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
              using T_ACC = acc_type<scalar_t, true>;
              const T_ACC dy_acc = static_cast<T_ACC>(dy);
              const T_ACC x_acc = static_cast<T_ACC>(x);
              const T_ACC s_acc =
                  T_ACC(1) / (T_ACC(1) + c10::cuda::compat::exp(-x_acc));
              return dy_acc * s_acc * (T_ACC(1) + x_acc * (T_ACC(1) - s_acc));
            });
      });
}

void mish_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mish_cuda",
      [&]() {
        gpu_kernel(
            iter,
            [] GPU_LAMBDA(scalar_t x) -> scalar_t {
          using T_ACC = acc_type<scalar_t, true>;
          const T_ACC x_acc = static_cast<T_ACC>(x);
          return x_acc * c10::cuda::compat::tanh(c10::cuda::compat::log1p(c10::cuda::compat::exp(x_acc)));
      });
      });
}

void mish_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mish_backward_cuda",
      [&]() {
        gpu_kernel(
            iter,
            [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
          using T_ACC = acc_type<scalar_t, true>;
          const T_ACC dy_acc = static_cast<T_ACC>(dy);
          const T_ACC x_acc = static_cast<T_ACC>(x);
          const T_ACC s_acc =
              T_ACC(1) / (T_ACC(1) + c10::cuda::compat::exp(-x_acc));
          const T_ACC t_acc =
              c10::cuda::compat::tanh(c10::cuda::compat::log1p(c10::cuda::compat::exp(x_acc)));
          return dy_acc * (t_acc + x_acc * s_acc * (T_ACC(1) - t_acc * t_acc));
      });
      });
}

} // namespace

TORCH_IMPL_FUNC(gelu_out_cuda) (
  const Tensor& self, const Tensor& result
) {
  GeluCUDAKernelImpl(*this);
}

TORCH_IMPL_FUNC(gelu_backward_out_cuda) (
  const Tensor& grad, const Tensor& self, const Tensor& grad_input
) {
  GeluBackwardCUDAKernelImpl(*this);
}

REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel);
REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel);
REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel);
REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel);
REGISTER_DISPATCH(elu_stub, &elu_kernel);
REGISTER_DISPATCH(elu_backward_stub, &elu_backward_kernel);
REGISTER_DISPATCH(leaky_relu_stub, &leaky_relu_kernel);
REGISTER_DISPATCH(leaky_relu_backward_stub, &leaky_relu_backward_kernel);
REGISTER_DISPATCH(hardswish_stub, &hardswish_kernel);
REGISTER_DISPATCH(hardswish_backward_stub, &hardswish_backward_kernel);
REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel);
REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel);
REGISTER_DISPATCH(softplus_stub, &softplus_kernel);
REGISTER_DISPATCH(softplus_backward_stub, &softplus_backward_kernel);
REGISTER_DISPATCH(silu_stub, &silu_kernel);
REGISTER_DISPATCH(silu_backward_stub, &silu_backward_kernel);
REGISTER_DISPATCH(mish_stub, &mish_kernel);
REGISTER_DISPATCH(mish_backward_stub, &mish_backward_kernel);
REGISTER_DISPATCH(threshold_stub, &threshold_kernel_cuda);

} // namespace native
} // namespace at
