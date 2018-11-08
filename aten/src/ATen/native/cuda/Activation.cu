#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/native/Activation.h"
#include "ATen/native/cuda/Loops.cuh"


namespace at { namespace native {

// -----------------------------------
// prelu forward
// -----------------------------------
template <typename scalar_t>
void prelu_cuda_kernel_share_weights(
  const Tensor& input,
  Tensor& result,
  const scalar_t* weight_data) {

  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
    input,
    result,
    [=] __device__ (
      const scalar_t& input_val,
      scalar_t& result_val) {
        result_val = (input_val > 0) ? input_val : *weight_data * input_val;
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
  AT_CHECK(self.is_cuda());
  AT_CHECK(weight_.is_cuda());

  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  AT_CHECK(input.is_contiguous());
  AT_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  Tensor result = at::empty_like(input);
  auto strides = input.strides();

  // case1: shared weight for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_cuda", [&] {
      prelu_cuda_kernel_share_weights<scalar_t>(
        input,
        result,
        weight.data<scalar_t>());
    });
  }
  else { // case2: multiple weights, one for each channel
    int64_t input_ndim = input.dim();
    AT_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    AT_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = %d, and channel size = %d.",
      weight_num, channel_size);

    // config to run cuda kernel
    int64_t input_numel = input.numel();
    const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), input_numel));
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    AT_CHECK(cuda::getApplyGrid(input_numel, grid, curDevice), "prelu: input too large or too many dimensions");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_cuda", [&] {
      prelu_cuda_kernel_multi_weights<scalar_t>
      <<<grid, block, 0, stream>>>(
        result.data<scalar_t>(),
        input.data<scalar_t>(),
        weight.data<scalar_t>(),
        input_stride0,
        input_stride1,
        input_numel);
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

  at::cuda::CUDA_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
    input,
    grad_out,
    input_grad,
    weight_grad_collector,
    [=] __device__ (
      const scalar_t& input_val,
      const scalar_t& grad_out_val,
      scalar_t& input_grad_val,
      scalar_t& weight_grad_collector_val) {
        input_grad_val = (input_val > 0) ? grad_out_val : *weight_data * grad_out_val;
        weight_grad_collector_val = (input_val > 0) ? scalar_t(0) : input_val * grad_out_val;
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
  AT_CHECK(grad_out_.is_cuda());
  AT_CHECK(self.is_cuda());
  AT_CHECK(weight_.is_cuda());

  auto input = self.contiguous();
  auto grad_out = grad_out_.contiguous();
  auto weight = weight_.contiguous();

  AT_CHECK(input.is_contiguous());
  AT_CHECK(weight.is_contiguous());
  AT_CHECK(grad_out.is_contiguous());

  int64_t weight_num = weight.numel();
  auto strides = input.strides();
  auto dims = input.dim();
  Tensor input_grad = at::empty_like(input);
  Tensor weight_grad = at::empty_like(weight);
  Tensor weight_grad_collector = at::empty_like(input);
  // case1: shared parameter for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_backward_cuda", [&] {
      prelu_cuda_backward_kernel_share_weights<scalar_t>(
        input,
        grad_out,
        input_grad,
        weight_grad_collector,
        weight.data<scalar_t>());
    });
    weight_grad.fill_(weight_grad_collector.sum());
  }
  else { // case2: multiple parameters, one for each channel
    int64_t input_ndim = input.dim();
    AT_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    AT_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = %d, and channel size = %d.",
      weight_num, channel_size);

    // config to run cuda kernel
    int64_t input_numel = input.numel();
    const dim3 block = dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), input_numel));
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
    AT_CHECK(cuda::getApplyGrid(input_numel, grid, curDevice), "prelu_backward_cuda: input too large or too many dimensions");

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "prelu_backward_cuda", [&] {
      prelu_cuda_backward_kernel_multi_weights<scalar_t>
      <<<grid, block, 0, stream>>>(
        input.data<scalar_t>(),
        weight.data<scalar_t>(),
        grad_out.data<scalar_t>(),
        input_grad.data<scalar_t>(),
        weight_grad_collector.data<scalar_t>(),
        input_stride0,
        input_stride1,
        input_numel);
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
// hardshrink
// -----------------------------------
template <typename scalar_t>
void hardshrink_cuda_kernel(const Tensor& self, Tensor& out_tensor, scalar_t lambd) {
  at::cuda::CUDA_tensor_apply2<scalar_t, scalar_t>(
    self,
    out_tensor,
    [=] __device__ (
      scalar_t& self_val,
      scalar_t& out_tensor_val) {
        out_tensor_val = (self_val >= -lambd && self_val <= lambd) ? scalar_t(0) : self_val;
  });
}

template <typename scalar_t>
void hardshrink_backward_cuda_kernel(const Tensor& self, Tensor& out_tensor, scalar_t lambd, const Tensor& grad) {
  at::cuda::CUDA_tensor_apply3<scalar_t, scalar_t, scalar_t>(
    self,
    grad,
    out_tensor,
    [=] __device__ (
      scalar_t& self_val,
      scalar_t& grad_val,
      scalar_t& out_tensor_val) {
        out_tensor_val = (self_val >= -lambd && self_val <= lambd) ? scalar_t(0) : grad_val;
  });
}

Tensor hardshrink_cuda(const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_cuda", [&] {
    hardshrink_cuda_kernel<scalar_t>(self, out_tensor, lambd.to<scalar_t>());
  });
  return out_tensor;
}

Tensor hardshrink_backward_cuda(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(grad);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "hardshrink_backward_cuda", [&] {
    hardshrink_backward_cuda_kernel<scalar_t>(self, out_tensor, lambd.to<scalar_t>(), grad);
  });
  return out_tensor;
}

template <typename scalar_t>
void threshold_kernel_impl(TensorIterator& iter, scalar_t threshold, scalar_t value) {
  gpu_binary_kernel(iter, [=]GPU_LAMBDA(scalar_t x, scalar_t other) -> scalar_t {
    return x <= threshold ? value : other;
  });
}

static void threshold_kernel(TensorIterator& iter, Scalar threshold, Scalar value) {
  AT_DISPATCH_ALL_TYPES_AND_HALF(iter.type(), "threshold", [&] {
    threshold_kernel_impl<scalar_t>(iter, threshold.to<scalar_t>(), value.to<scalar_t>());
  });
}

REGISTER_DISPATCH(threshold_stub, &threshold_kernel);

}}  // namespace at::native
