#include "ATen/ATen.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"
#include "ATen/Utils.h"
#include "c10/util/Exception.h"
#include <THC/THCGeneral.h>
#include "THC/THCNumerics.cuh"

#include <cmath>

namespace at {
namespace native {
namespace {

using at::cuda::detail::canUse32BitIndexMath;

__device__
inline void get_index_mapping(
    int64_t input_dim_x, int64_t output_x, int64_t pad_l,
    int64_t & input_idx, int64_t & output_idx) {
  // 3D grid of 1D blocks
  auto input_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * input_dim_x;
  auto output_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * gridDim.x * blockDim.x;

  auto i_start_x = ::max(0L, -pad_l);
  auto o_start_x = ::max(0L, pad_l);

  int64_t input_x = ::abs(output_x - pad_l)
                    - ::abs(output_x - (input_dim_x + pad_l - 1))
                    - output_x
                    + 2 * pad_l + input_dim_x - 1
                    - o_start_x + i_start_x;

  input_idx = input_offset + input_x;
  output_idx = output_offset + output_x;
}

template<typename scalar_t>
__global__ void reflection_pad1d_out_kernel(
    scalar_t * input, scalar_t * output,
    int64_t input_dim_x,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;

  if (output_x < output_dim_x) {
    int64_t input_idx, output_idx;
    get_index_mapping(input_dim_x, output_x, pad_l, input_idx, output_idx);

    output[output_idx] = input[input_idx];
  }
}

template <typename scalar_t>
__global__ void reflection_pad1d_backward_out_kernel(
    scalar_t * grad_input, scalar_t * grad_output,
    int64_t input_dim_x,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;

  if (output_x < output_dim_x) {
    int64_t input_idx, output_idx;
    get_index_mapping(input_dim_x, output_x, pad_l, input_idx, output_idx);

    atomicAdd(&grad_input[input_idx], grad_output[output_idx]);
  }
}


void reflection_pad1d_out_template(
    Tensor &output, const Tensor &input_, IntList padding) {
  AT_CHECK(canUse32BitIndexMath(input_),
    "input tensor must fit into 32-bit index math");

  int plane_dim = 0;
  int dimw = 1;
  int num_batch = 1;

  for (int64_t i = 0; i < input_.ndimension(); ++i) {
    AT_CHECK(input_.size(i) > 0,
      "reflection_pad1d(): expected input to have non-empty temporal "
      "dimensions, but input has sizes ", input_.sizes(), "with dimension ", i,
      " being empty");
  }

  AT_CHECK(input_.ndimension() == 2 || input_.ndimension() == 3, "non-empty 2D "
    "or 3D (batch mode) tensor expected for input, but got: ", input_);

  if (input_.ndimension() == 3) {
    num_batch = input_.size(0);
    plane_dim++;
    dimw++;
  }

  int num_planes = input_.size(plane_dim);
  int input_w = input_.size(dimw);

  int pad_l = padding[0];
  int pad_r = padding[1];

  AT_CHECK(pad_l < input_w && pad_r < input_w, "Padding size should be less "
    "than the corresponding input dimension, but got: padding (",  pad_l, ", ",
    pad_r, ") at dimension ", dimw, " of input ", input_);

  int output_w  = input_w + pad_l + pad_r;

  AT_CHECK(output_w >= 1,
    "input (W: ", input_w, ")is too small. Calculated output W: ", output_w);

  if (input_.ndimension() == 2) {
    output.resize_({num_planes, output_w});
  } else {
    output.resize_({num_batch, num_planes, output_w});
  }

  int output_plane_size = output.size(2);
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);
  dim3 grid_size(
    (int) ::ceil(output_plane_size / 256.0), output.size(1), output.size(0));

  Tensor input = input_.contiguous();
  AT_DISPATCH_ALL_TYPES_AND_HALF(
    input.type(), "reflection_pad1d_out_template", [&] {
      scalar_t * input_data = input.data<scalar_t>();
      scalar_t * output_data = output.data<scalar_t>();

      reflection_pad1d_out_kernel<<<
        grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
          input_data, output_data, input.size(2), pad_l, pad_r);
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

  int plane_dim = 0;
  int dimw = 1;

  if (input.ndimension() == 3) {
    plane_dim++;
    dimw++;
  }

  auto pad_l = padding[0];
  auto pad_r = padding[1];
  int iwidth = input.size(dimw);
  int owidth  = iwidth + pad_l + pad_r;

  Tensor grad_output = grad_output_.contiguous();

  AT_CHECK(owidth == grad_output.size(dimw),
    "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
    grad_output.size(dimw));

  int grad_output_plane_size = grad_output.size(2);
  dim3 block_size(
    grad_output_plane_size > 256 ? 256 : grad_output_plane_size);
  dim3 grid_size(
    (int) ::ceil(grad_output_plane_size / 256.0),
    grad_output.size(1),
    grad_output.size(0));

  AT_DISPATCH_ALL_TYPES_AND_HALF(
    input.type(), "reflection_pad1d_backward_out_template", [&] {
      scalar_t * grad_input_data = grad_input.data<scalar_t>();
      scalar_t * grad_output_data = grad_output.data<scalar_t>();

      reflection_pad1d_backward_out_kernel<<<
        grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_input_data, grad_output_data, grad_input.size(2), pad_l, pad_r);
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

Tensor reflection_pad1d_cuda(Tensor const& input, IntList padding) {
  auto output = at::empty({0}, input.options());
  reflection_pad1d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad1d_backward_out_cuda(
    Tensor& grad_input, const Tensor& grad_output,
    const Tensor& input,
    IntList padding) {
  grad_input.resize_as_(input);
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
