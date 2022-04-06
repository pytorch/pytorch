#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/core/Scalar.h>

namespace at {
namespace native {

// -----------------------------------
// glu forward
// -----------------------------------
void glu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "glu_cuda", [&]() {
    using acc_t = at::acc_type<scalar_t, true>;
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
      const acc_t a = a_;
      const acc_t b = b_;
      const acc_t one = acc_t(1);
      const acc_t sigmoid = one / (one + std::exp(-b));
      return a * sigmoid;
    });
  });
}

// -----------------------------------
// glu backward
// -----------------------------------

// Byte offsets don't require multiplication by sizeof(T), so are slightly cheaper.
// For fixed offsets, this removes all penalty from 64-bit indexing.
template <typename T>
__device__ T* byte_offset(T* ptr, int64_t offset) {
  using byte_ptr_t = typename std::conditional<
    std::is_const<T>::value, const char*, char*>::type;
  return reinterpret_cast<T*>(
    reinterpret_cast<byte_ptr_t>(ptr) + offset
  );
}

template <typename scalar_t, typename OffsetCalc>
__global__ void glu_backward_kernel(
    int numel, scalar_t* gI, const scalar_t* I, const scalar_t* gO,
    OffsetCalc offset_calculator,
    int64_t gI_byte_offset, int64_t I_byte_offset) {
  using acc_t = at::acc_type<scalar_t, true>;

  const uint32_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_index >= numel) {
    return;
  }
  const auto offsets = offset_calculator.get(linear_index);

  // We explicitly iterate over the first half of the input tensor, and
  // gI_byte_offset and I_byte_offset are the offsets to access the
  // corresponding index in the second half of the tensor.
  const acc_t a = I[offsets[1]];
  const acc_t b = *byte_offset(I + offsets[1], I_byte_offset);
  const acc_t gO_val = gO[offsets[2]];

  const auto one = acc_t(1);
  const acc_t sigmoid = one / (one + std::exp(-b));

  auto* gA = gI + offsets[0];
  *gA = sigmoid * gO_val;

  auto* gB = byte_offset(gA, gI_byte_offset);
  *gB = (one - sigmoid) * sigmoid * gO_val * a;
}

void launch_glu_backward_kernel(const TensorIteratorBase& iter,
                                int64_t gI_stride, int64_t I_stride) {
  const auto N = iter.numel();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(N > 0 && N <= std::numeric_limits<int32_t>::max());
  const auto offset_calculator = make_element_offset_calculator<3>(iter);
  constexpr int64_t block_size = 256;
  const int64_t grid = (N + block_size - 1) / block_size;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "glu_backward_cuda", [&] {
    auto gI = static_cast<scalar_t*>(iter.data_ptr(0));
    auto I = static_cast<const scalar_t*>(iter.data_ptr(1));
    auto gO = static_cast<const scalar_t*>(iter.data_ptr(2));
    glu_backward_kernel<<<grid, block_size, 0, stream>>>(
        N, gI, I, gO, offset_calculator,
        gI_stride * sizeof(scalar_t), I_stride * sizeof(scalar_t));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
}

// -----------------------------------
// log_sigmoid forward
// -----------------------------------

void launch_log_sigmoid_forward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.common_dtype(),
                                 "log_sigmoid_forward_cuda", [&] {
    using acc_t = acc_type<scalar_t, true>;
    gpu_kernel(iter,
        [] GPU_LAMBDA (scalar_t in_) -> scalar_t {
          const acc_t in = in_;
          const auto min = std::min(acc_t(0), in);
          const auto z = std::exp(-std::abs(in));
          return min - std::log1p(z);
        });
  });
}

// -----------------------------------
// log_sigmoid backward
// -----------------------------------

void log_sigmoid_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.common_dtype(),
                                 "log_sigmoid_backward_cuda", [&] {
    using acc_t = acc_type<scalar_t, true>;
    gpu_kernel(iter,
        [] GPU_LAMBDA (scalar_t in_, scalar_t grad_out_) -> scalar_t {
          const acc_t in = in_;
          const acc_t grad_out = grad_out_;

          auto in_negative = in < acc_t(0);
          auto max_deriv = in_negative ? acc_t(1) : acc_t(0);
          auto sign = in_negative ? acc_t(1) : -acc_t(1);
          const auto z = std::exp(-std::abs(in));
          return grad_out * (max_deriv - sign * (z / (acc_t(1) + z)));
        });
  });
}

// -----------------------------------
// prelu forward
// -----------------------------------
void launch_prelu_cuda_kernel_share_weights(TensorIteratorBase &iter, const TensorBase &weight) {
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, iter.input_dtype(), "prelu_cuda", [&] {
    const auto *weight_data = weight.data_ptr<scalar_t>();
    at::native::gpu_kernel(iter,
        [weight_data] GPU_LAMBDA (scalar_t input_val) {
          return (input_val > 0) ? input_val : *weight_data * input_val;
        });
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

void launch_prelu_cuda_kernel_multi_weights(
    const TensorBase &result, const TensorBase &input, const TensorBase &weight) {
  int64_t input_ndim = input.dim();
  TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

  int64_t channel_size = 1; // channel_size default to 1
  int64_t input_stride0 = 1, input_stride1 = 1;

  if (input_ndim > 1) {
    channel_size = input.size(1); // channel is the 2nd dim of input
    auto strides = input.strides();
    input_stride0 = strides[0];
    input_stride1 = strides[1];
  }
  const int64_t weight_num = weight.numel();
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

// -----------------------------------
// prelu backward
// -----------------------------------
void launch_prelu_cuda_backward_kernel_share_weights(
    TensorIteratorBase &iter, const TensorBase &weight) {
  // N.B. `std::tuple` does not support `::operator=` on device code.
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Half, iter.input_dtype(), "prelu_backward_cuda", [&] {
    const auto *weight_data = weight.data_ptr<scalar_t>();
    gpu_kernel_multiple_outputs(iter, [=] GPU_LAMBDA (scalar_t input, scalar_t grad_out) -> thrust::tuple<scalar_t, scalar_t> {
        scalar_t input_grad = input > 0 ? grad_out : (*weight_data) * grad_out;
        scalar_t weight_grad_collector = input > 0 ? scalar_t(0) : input * grad_out;
        return {input_grad, weight_grad_collector};
      });
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

void launch_prelu_cuda_backward_kernel_multi_weights(
    const TensorBase &input, const TensorBase &weight, const TensorBase &grad_out,
    const TensorBase &input_grad, const TensorBase &weight_grad_collector) {
  int64_t input_ndim = input.dim();
  TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

  int64_t channel_size = 1; // channel_size default to 1
  int64_t input_stride0 = 1, input_stride1 = 1;

  if (input_ndim > 1) {
    channel_size = input.size(1); // channel is the 2nd dim of input
    auto strides = input.strides();
    input_stride0 = strides[0];
    input_stride1 = strides[1];
  }
  const int64_t weight_num = weight.numel();
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

void GeluCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, it.dtype(), "GeluCUDAKernelImpl", [&]() {
      gpu_kernel(it, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
        constexpr opmath_t kKappa = 0.044715;
        auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
        auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
        return opmath_t(0.5) * static_cast<opmath_t>(x) * (opmath_t(1) + c10::cuda::compat::tanh(inner));
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, it.dtype(), "GeluCUDAKernelImpl", [&]() {
      gpu_kernel(it, [] GPU_LAMBDA(scalar_t x) -> scalar_t {
        using opmath_t = at::opmath_type<scalar_t>;
        constexpr opmath_t kAlpha = M_SQRT1_2;
        return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
      });
    });
  }
}

void GeluBackwardCUDAKernelImpl(TensorIteratorBase& it, GeluType approximate) {
  if (approximate == GeluType::Tanh) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        it.dtype(), "GeluBackwardCUDAKernelImpl", [&]() {
          gpu_kernel(it, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            constexpr opmath_t kBeta = M_SQRT2 * M_2_SQRTPI * opmath_t(0.5);
            constexpr opmath_t kKappa = 0.044715;
            auto x_sq = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);
            auto x_cube = x_sq * static_cast<opmath_t>(x);
            auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);
            auto tanh_inner = c10::cuda::compat::tanh(inner);

            auto left = opmath_t(0.5) * static_cast<opmath_t>(x);
            auto right = opmath_t(1) + tanh_inner;

            auto left_derivative = 0.5 * right;

            auto tanh_derivative = opmath_t(1) - tanh_inner * tanh_inner;
            auto inner_derivative = kBeta * (opmath_t(1) + opmath_t(3) * kKappa * x_sq);
            auto right_derivative = left * tanh_derivative * inner_derivative;

            return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);
        });
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        it.dtype(), "GeluBackwardCUDAKernelImpl", [&]() {
          gpu_kernel(it, [] GPU_LAMBDA(scalar_t dy, scalar_t x) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
            constexpr opmath_t kAlpha = M_SQRT1_2;
            const opmath_t cdf =
                opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
            const opmath_t pdf =
                c10::cuda::compat::exp(
                    opmath_t(-0.5) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x)) *
                kBeta;
            return static_cast<opmath_t>(dy) * (cdf + static_cast<opmath_t>(x) * pdf);
          });
        });
  }
}

namespace {

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

REGISTER_DISPATCH(hardtanh_backward_stub, &hardtanh_backward_kernel);
REGISTER_DISPATCH(hardshrink_stub, &hardshrink_kernel);
REGISTER_DISPATCH(log_sigmoid_backward_stub, &log_sigmoid_backward_kernel);
REGISTER_DISPATCH(softshrink_stub, &softshrink_kernel);
REGISTER_DISPATCH(shrink_backward_stub, &shrink_backward_kernel);
REGISTER_DISPATCH(elu_stub, &elu_kernel);
REGISTER_DISPATCH(elu_backward_stub, &elu_backward_kernel);
REGISTER_DISPATCH(glu_stub, &glu_kernel);
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
