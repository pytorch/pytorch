#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
// keeping THC headers for atomicAdd
#include <THC/THCAtomics.cuh>

#include <thrust/pair.h>

namespace at {
namespace native {
namespace {

using at::cuda::detail::canUse32BitIndexMath;

__device__
inline thrust::pair<int64_t, int64_t> get_index_mapping(
    int64_t input_w, int64_t output_w,
    int64_t output_x,
    int64_t pad_l) {
  // 3D grid of 1D blocks
  auto input_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * input_w;
  auto output_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * output_w;

  auto i_start_x = ::max(int64_t(0), -pad_l);
  auto o_start_x = ::max(int64_t(0), pad_l);

  int64_t input_x = ::abs(output_x - pad_l)
                    - ::abs(output_x - (input_w + pad_l - 1))
                    - output_x
                    + 2 * pad_l + input_w - 1
                    - o_start_x + i_start_x;

  return thrust::make_pair<int64_t, int64_t>(
    input_offset + input_x, output_offset + output_x);
}

template<typename scalar_t>
__global__ void reflection_pad1d_out_kernel(
    scalar_t * input, scalar_t * output,
    int64_t input_w,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_w = input_w + pad_l + pad_r;

  if (output_x < output_w) {
    auto index_pair = get_index_mapping(input_w, output_w, output_x, pad_l);
    output[index_pair.second] = input[index_pair.first];
  }
}

template <typename scalar_t>
__global__ void reflection_pad1d_backward_out_kernel(
    scalar_t * grad_input, scalar_t * grad_output,
    int64_t input_w,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_w = input_w + pad_l + pad_r;

  if (output_x < output_w) {
    auto index_pair = get_index_mapping(input_w, output_w, output_x, pad_l);
    atomicAdd(
      &grad_input[index_pair.first], grad_output[index_pair.second]);
  }
}

void reflection_pad1d_out_template(
    Tensor &output, const Tensor &input_, IntList padding) {
  AT_CHECK(canUse32BitIndexMath(input_),
    "input tensor must fit into 32-bit index math");

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  AT_CHECK(input_.numel() > 0 &&
    (input_.ndimension() == 2 || input_.ndimension() == 3), "non-empty 2D "
    "or 3D (batch mode) tensor expected for input, but got: ", input_);

  if (input_.ndimension() == 3) {
    nbatch = input_.size(0);
    dim_plane++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  int64_t nplane = input_.size(dim_plane);
  int64_t input_w = input_.size(dim_w);
  int64_t output_w  = input_w + pad_l + pad_r;

  AT_CHECK(pad_l < input_w && pad_r < input_w, "Padding size should be less "
    "than the corresponding input dimension, but got: padding (",  pad_l, ", ",
    pad_r, ") at dimension ", dim_w, " of input ", input_);

  AT_CHECK(output_w >= 1,
    "input (W: ", input_w, ")is too small. Calculated output W: ", output_w);

  if (input_.ndimension() == 2) {
    output.resize_({nplane, output_w});
  } else {
    output.resize_({nbatch, nplane, output_w});
  }

  dim3 block_size(output_w > 256 ? 256 : output_w);
  dim3 grid_size((int) ::ceil(output_w / 256.0), nplane, nbatch);

  Tensor input = input_.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.type(), "reflection_pad1d_out_template", [&] {
      reflection_pad1d_out_kernel<<<
        grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
          input.data<scalar_t>(), output.data<scalar_t>(),
          input_w, pad_l, pad_r);
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());
}

void reflection_pad1d_backward_out_template(
    Tensor & grad_input, const Tensor & grad_output_,
    const Tensor & input, IntList padding) {

  AT_CHECK(canUse32BitIndexMath(input),
    "input tensor must fit into 32-bit index math");

  AT_CHECK(canUse32BitIndexMath(grad_output_),
    "input tensor must fit into 32-bit index math");

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dim_plane++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  int64_t nplane = input.size(dim_plane);
  int64_t input_w = input.size(dim_w);
  int64_t output_w  = input_w + pad_l + pad_r;

  Tensor grad_output = grad_output_.contiguous();

  AT_CHECK(output_w == grad_output.size(dim_w),
    "gradOutput width unexpected. Expected: ", output_w, ", Got: ",
    grad_output.size(dim_w));

  dim3 block_size(output_w > 256 ? 256 : output_w);
  dim3 grid_size((int) ::ceil(output_w / 256.0), nplane, nbatch);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad_input.type(), "reflection_pad1d_backward_out_template", [&] {
      reflection_pad1d_backward_out_kernel<<<
        grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_input.data<scalar_t>(), grad_output.data<scalar_t>(),
          input_w, pad_l, pad_r);
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

Tensor& reflection_pad1d_out_cuda(
    Tensor& output, const Tensor& input, IntList padding) {
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad1d_cuda(const Tensor& input, IntList padding) {
  auto output = at::empty({0}, input.options());
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad1d_backward_out_cuda(
    Tensor& grad_input, const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  reflection_pad1d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad1d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  auto grad_input = at::zeros_like(input);
  reflection_pad1d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace native
} // namespace at
