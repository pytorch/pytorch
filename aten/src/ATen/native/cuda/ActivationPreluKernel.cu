#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at {
namespace native {

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

} // namespace native
} // namespace at
