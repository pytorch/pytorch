#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/Padding.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/reflection_pad1d_native.h>
#include <ATen/ops/reflection_pad2d_native.h>
#include <ATen/ops/reflection_pad3d_native.h>
#include <ATen/ops/reflection_pad1d_backward_native.h>
#include <ATen/ops/reflection_pad2d_backward_native.h>
#include <ATen/ops/reflection_pad3d_backward_native.h>
#endif

#include <thrust/pair.h>

namespace at::native {
namespace {

using at::cuda::detail::canUse32BitIndexMath;

__device__
inline thrust::pair<int64_t, int64_t> get_index_mapping1d(
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


__device__
inline thrust::pair<int64_t, int64_t>  get_index_mapping2d(
    int64_t input_dim_x, int64_t input_dim_y,
    int64_t output_dim_x, int64_t output_dim_y,
    int64_t pad_l, int64_t pad_t,
    int64_t output_xy, int y_shift, int z_shift, int nplane) {
  // 3D grid of 1D blocks
  auto input_offset =
    ((blockIdx.y + y_shift) + (blockIdx.z + z_shift) * nplane) * input_dim_x * input_dim_y;
  auto output_offset =
    ((blockIdx.y + y_shift) + (blockIdx.z + z_shift) * nplane) * output_dim_x * output_dim_y;

  auto output_x = output_xy % output_dim_x;
  auto output_y = output_xy / output_dim_x;

  auto i_start_x = ::max(int64_t(0), -pad_l);
  auto i_start_y = ::max(int64_t(0), -pad_t);
  auto o_start_x = ::max(int64_t(0), pad_l);
  auto o_start_y = ::max(int64_t(0), pad_t);

  auto input_x = ::abs(output_x - pad_l)
                 - ::abs(output_x - (input_dim_x + pad_l - 1))
                 - output_x
                 + 2 * pad_l + input_dim_x - 1
                 - o_start_x + i_start_x;

  auto input_y = ::abs(output_y - pad_t)
                 - ::abs(output_y - (input_dim_y + pad_t - 1))
                 - output_y
                 + 2 * pad_t + input_dim_y - 1
                 - o_start_y + i_start_y;

  return thrust::make_pair<int64_t, int64_t>(
    input_offset + input_y * input_dim_x + input_x,
    output_offset + output_y * output_dim_x + output_x);
}

template<typename scalar_t>
__global__ void reflection_pad1d_out_kernel(
    const scalar_t * input, scalar_t * output,
    int64_t input_w,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_w = input_w + pad_l + pad_r;

  if (output_x < output_w) {
    auto index_pair = get_index_mapping1d(input_w, output_w, output_x, pad_l);
    output[index_pair.second] = input[index_pair.first];
  }
}

template <typename scalar_t>
__global__ void reflection_pad1d_backward_out_kernel(
    scalar_t * grad_input, const scalar_t * grad_output,
    int64_t input_w,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_w = input_w + pad_l + pad_r;

  if (output_x < output_w) {
    auto index_pair = get_index_mapping1d(input_w, output_w, output_x, pad_l);
    gpuAtomicAddNoReturn(
      &grad_input[index_pair.first], grad_output[index_pair.second]);
  }
}

template<typename scalar_t>
__global__ void reflection_pad2d_out_kernel(
    const scalar_t * input, scalar_t * output,
    int64_t input_dim_x, int64_t input_dim_y,
    int pad_t, int pad_b, int pad_l, int pad_r, int y_shift, int z_shift, int nplane) {
  auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;
  auto output_dim_y = input_dim_y + pad_t + pad_b;

  if (output_xy < output_dim_x * output_dim_y) {
    auto index_pair = get_index_mapping2d(
      input_dim_x, input_dim_y,
      output_dim_x, output_dim_y,
      pad_l, pad_t,
      output_xy, y_shift, z_shift, nplane);

    output[index_pair.second] = input[index_pair.first];
  }
}

template <typename scalar_t>
__global__ void reflection_pad2d_backward_out_kernel(
    scalar_t * grad_input, const scalar_t * grad_output,
    int64_t input_dim_x, int64_t input_dim_y,
    int pad_t, int pad_b, int pad_l, int pad_r, int y_shift, int z_shift, int nplane) {
  auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;
  auto output_dim_y = input_dim_y + pad_t + pad_b;

  if (output_xy < output_dim_x * output_dim_y) {
    auto index_pair = get_index_mapping2d(
      input_dim_x, input_dim_y,
      output_dim_x, output_dim_y,
      pad_l, pad_t,
      output_xy, y_shift, z_shift, nplane);

    gpuAtomicAddNoReturn(&grad_input[index_pair.first], grad_output[index_pair.second]);
  }
}

template <typename scalar_t>
__global__ void reflection_pad2d_backward_det_out_kernel(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    int64_t input_dim_x,
    int64_t input_dim_y,
    int pad_t,
    int pad_b,
    int pad_l,
    int pad_r,
    int batch,
    int channels,
    int) {
  const int64_t input_xy_ = threadIdx.x + blockIdx.x * blockDim.x;
  const auto output_dim_x = input_dim_x + pad_l + pad_r;
  const auto output_dim_y = input_dim_y + pad_t + pad_b;
  const auto N = output_dim_x * output_dim_y;
  const int64_t width = output_dim_x;
  const int64_t height = output_dim_y;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  const int64_t end =
      static_cast<int64_t>(batch) * channels * input_dim_x * input_dim_y;

  for (int64_t input_xy = input_xy_; input_xy < end; input_xy += stride) {
    scalar_t partial = 0;

    const int64_t b = input_xy / (channels * input_dim_x * input_dim_y);
    const int64_t c = (input_xy / (input_dim_x * input_dim_y)) % channels;
    const int64_t pos_xy = input_xy % (input_dim_x * input_dim_y);
    const int64_t inp_row = pos_xy / input_dim_x;
    const int64_t inp_col = pos_xy % input_dim_x;

    const bool is_top = (inp_row >= 1) && (inp_row <= pad_t);
    const bool is_bottom =
        (inp_row < input_dim_y - 1) && (inp_row >= input_dim_y - pad_b - 1);
    const bool is_left = (inp_col >= 1) && (inp_col <= pad_l);
    const bool is_right =
        (inp_col < input_dim_x - 1) && (inp_col >= input_dim_x - pad_r - 1);

    if (is_top) {
      const int64_t border_top_row = 0;
      const int64_t dist_from_t = inp_row;

      const int64_t border_top_out_row = border_top_row + pad_t;
      const int64_t border_top_out_col = pad_l + inp_col;

      const int64_t reflected_top_row = border_top_out_row - dist_from_t;
      const int64_t reflected_top_out =
          reflected_top_row * width + border_top_out_col;

      if (reflected_top_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflected_top_out];
      }

      if (is_left) { // top left
        const int64_t corner_tl_out_row = pad_t;
        const int64_t corner_tl_out_col = pad_l;
        const int64_t dist_rows = inp_row;
        const int64_t dist_cols = inp_col;
        const int64_t reflect_tl_out_row = (corner_tl_out_row - dist_rows);
        const int64_t reflect_tl_out_col = (corner_tl_out_col - dist_cols);
        const int64_t reflect_tl_out =
            (reflect_tl_out_row * width) + reflect_tl_out_col;

        if (reflect_tl_out >= 0 && reflect_tl_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_tl_out];
        }
      } else if (is_right) { // top right
        // TR corner is just (0, cols - 1)
        const int64_t corner_tr_out_row = pad_t;
        const int64_t corner_tr_out_col = pad_l + input_dim_x - 1;
        const int64_t dist_rows = inp_row; // as the TR corner is (0, cols - 1)
        const int64_t dist_cols = ::abs(inp_col - (input_dim_x - 1));

        // we were dist_rows after, now we want to be dist_rows before
        // we were dist_cols before, now we wnat to be dist_cols after
        const int64_t reflect_tr_out_row = (corner_tr_out_row - dist_rows);
        const int64_t reflect_tr_out_col = (corner_tr_out_col + dist_cols);
        const int64_t reflect_tr_out =
            (reflect_tr_out_row * width) + reflect_tr_out_col;

        if (reflect_tr_out >= 0 && reflect_tr_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_tr_out];
        }
      }
    }

    if (is_bottom) {
      const int64_t border_bot_row =
          input_dim_y - 1; // must use last row, not inp row
      const int64_t border_bot_col = inp_col;
      const int64_t dist_from_bot = ::abs(inp_row - border_bot_row);

      // we are dist_from_bot rows before it. Now we want to be after it.
      const int64_t border_bot_out_row = pad_t + border_bot_row;
      const int64_t border_bot_out_col = pad_l + border_bot_col;
      const int64_t reflect_bot_row = (border_bot_out_row + dist_from_bot);
      const int64_t reflect_bot_out =
          (reflect_bot_row * width) + border_bot_out_col;

      if (reflect_bot_out >= 0 && reflect_bot_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflect_bot_out];
      }

      if (is_left) {
        // (rows - 1, 0)
        const int64_t corner_bl_row = input_dim_y - 1;
        const int64_t corner_bl_col = 0;

        const int64_t corner_bl_out_row = pad_t + corner_bl_row;
        const int64_t corner_bl_out_col = pad_l + corner_bl_col;

        // we are inp_rows before it. inp_cols after it.
        const int64_t dist_rows = ::abs(inp_row - corner_bl_row);
        const int64_t dist_cols = inp_col;

        // Now we want to be inp_rows after, and inp_cols before.
        const int64_t reflect_bl_out_row = (corner_bl_out_row + dist_rows);
        const int64_t reflect_bl_out_col = (corner_bl_out_col - dist_cols);
        const int64_t reflect_bl_out =
            (reflect_bl_out_row * width) + reflect_bl_out_col;

        if (reflect_bl_out >= 0 && reflect_bl_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_bl_out];
        }
      } else if (is_right) {
        // (rows-1, cols-1)
        const int64_t corner_br_row = input_dim_y - 1;
        const int64_t corner_br_col = input_dim_x - 1;
        const int64_t dist_rows = ::abs(inp_row - corner_br_row);
        const int64_t dist_cols = ::abs(inp_col - corner_br_col);

        const int64_t corner_br_out_row = pad_t + corner_br_row;
        const int64_t corner_br_out_col = pad_l + corner_br_col;

        const int64_t reflect_br_out_row = (corner_br_out_row + dist_rows);
        const int64_t reflect_br_out_col = (corner_br_out_col + dist_cols);
        const int64_t reflect_br_out =
            (reflect_br_out_row * width) + reflect_br_out_col;

        if (reflect_br_out >= 0 && reflect_br_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_br_out];
        }
      }
    }
    if (is_left) {
      const int64_t border_left_row = inp_row;
      const int64_t border_left_out_row = border_left_row + pad_t;
      const int64_t border_left_out_col = pad_l;

      const int64_t dist_from_left = inp_col;

      const int64_t reflect_left_out_row = border_left_out_row;
      const int64_t reflect_left_out_col = border_left_out_col - dist_from_left;
      const int64_t reflect_left_out =
          reflect_left_out_row * width + reflect_left_out_col;

      if (reflect_left_out >= 0 && reflect_left_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflect_left_out];
      }
    }
    if (is_right) {
      const int64_t border_right_row = inp_row;
      const int64_t border_right_col = input_dim_x - 1;

      const int64_t border_right_out_row = border_right_row + pad_t;
      const int64_t border_right_out_col = border_right_col + pad_l;

      const int64_t dist_from_right = ::abs(inp_col - border_right_col);

      const int64_t reflect_right_out_row = border_right_out_row;
      const int64_t reflect_right_out_col =
          border_right_out_col + dist_from_right;
      const int64_t reflect_right_out =
          reflect_right_out_row * width + reflect_right_out_col;

      if (reflect_right_out >= 0 && reflect_right_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflect_right_out];
      }
    }
    const int64_t out_row = inp_row + pad_t;
    const int64_t out_col = inp_col + pad_l;

    partial += grad_output
        [b * (channels * width * height) + c * (width * height) +
         out_row * width + out_col];

    grad_input[input_xy] += partial;
  }
}

template <typename input_scalar_t, typename output_scalar_t, typename F>
__device__ inline void parallel_reflection_pad3d(
    PackedTensorAccessor64<input_scalar_t, 5> input,
    PackedTensorAccessor64<output_scalar_t, 5> output,
    int64_t pad_left,
    int64_t pad_top,
    int64_t pad_front,
    int64_t y_shift,
    int64_t z_shift,
    const F& f) {
  int64_t output_id = threadIdx.x + blockIdx.x * blockDim.x;

  if (output_id >= (output.size(2) * output.size(3) * output.size(4))) {
    return;
  }

  int64_t output_x = output_id % output.size(4);
  int64_t output_y = (output_id / output.size(4)) % output.size(3);
  int64_t output_z = output_id / (output.size(3) * output.size(4));

  int64_t i_start_x = ::max(int64_t(0), -pad_left);
  int64_t o_start_x = ::max(int64_t(0), pad_left);
  int64_t i_start_y = ::max(int64_t(0), -pad_top);
  int64_t o_start_y = ::max(int64_t(0), pad_top);
  int64_t i_start_z = ::max(int64_t(0), -pad_front);
  int64_t o_start_z = ::max(int64_t(0), pad_front);

  int64_t input_x = ::abs(output_x - pad_left)
                 - ::abs(output_x - (input.size(4) + pad_left - 1))
                 - output_x
                 + 2 * pad_left + input.size(4) - 1
                 - o_start_x + i_start_x;
  int64_t input_y = ::abs(output_y - pad_top)
                 - ::abs(output_y - (input.size(3) + pad_top - 1))
                 - output_y
                 + 2 * pad_top + input.size(3) - 1
                 - o_start_y + i_start_y;

  int64_t input_z = ::abs(output_z - pad_front)
                 - ::abs(output_z - (input.size(2) + pad_front - 1))
                 - output_z
                 + 2 * pad_front + input.size(2) - 1
                 - o_start_z + i_start_z;

  int64_t plane = blockIdx.y + y_shift;
  int64_t batch = blockIdx.z + z_shift;
  f(plane, batch, output_z, output_y, output_x, input_z, input_y, input_x);
}

template<typename scalar_t>
__global__ void reflection_pad3d_out_kernel(
    PackedTensorAccessor64<const scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
    int64_t pad_left,  int64_t pad_top, int64_t pad_front,
    int64_t y_shift, int64_t z_shift
){
  parallel_reflection_pad3d(
      input,
      output,
      pad_left,
      pad_top,
      pad_front,
      y_shift,
      z_shift,
      [&] __device__(
          int64_t plane,
          int64_t batch,
          int64_t output_z,
          int64_t output_y,
          int64_t output_x,
          int64_t input_z,
          int64_t input_y,
          int64_t input_x) {
        auto value_to_copy = input[batch][plane][input_z][input_y][input_x];
        output[batch][plane][output_z][output_y][output_x] = value_to_copy;
      });
}

template <typename scalar_t>
__global__ void reflection_pad3d_backward_out_kernel(
    PackedTensorAccessor64<scalar_t, 5> grad_input,
    PackedTensorAccessor64<const scalar_t, 5> grad_output,
    int64_t pad_left,  int64_t pad_top, int64_t pad_front,
    int64_t y_shift, int64_t z_shift
) {
  parallel_reflection_pad3d(
      grad_input,
      grad_output,
      pad_left,
      pad_top,
      pad_front,
      y_shift,
      z_shift,
      [&] __device__(
          int64_t plane,
          int64_t batch,
          int64_t output_z,
          int64_t output_y,
          int64_t output_x,
          int64_t input_z,
          int64_t input_y,
          int64_t input_x) {
        auto value_to_add = grad_output[batch][plane][output_z][output_y][output_x];
        auto target = &grad_input[batch][plane][input_z][input_y][input_x];
        gpuAtomicAddNoReturn(target, value_to_add);
      });
}

void reflection_pad2d_out_template(
    Tensor &output, const Tensor &input_, IntArrayRef padding) {

  TORCH_CHECK(canUse32BitIndexMath(input_),
    "input tensor must fit into 32-bit index math");

  int plane_dim = 0;
  int dim_h = 1;
  int dim_w = 2;
  int nbatch = 1;

  at::native::padding::check_valid_input<2>(input_, padding);

  if (input_.ndimension() == 4) {
    nbatch = input_.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int nplane = input_.size(plane_dim);
  int input_h = input_.size(dim_h);
  int input_w = input_.size(dim_w);

  TORCH_CHECK(pad_l < input_w && pad_r < input_w,
    "Padding size should be less than the corresponding input dimension, but "
    "got: padding (", pad_l, ", ", pad_r, ") at dimension ", dim_w,
    " of input ", input_.sizes());

  TORCH_CHECK(pad_t < input_h && pad_b < input_h,
    "Padding size should be less than the corresponding input dimension, but "
    "got: padding (", pad_t, ", ", pad_b, ") at dimension ", dim_h,
    " of input ", input_.sizes());

  int output_h = input_h + pad_t + pad_b;
  int output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(output_w >= 1 || output_h >= 1,
    "input (H: ", input_h, ", W: ", input_w, ") is too small.  Calculated "
    "output H: ", output_h, " W: ", output_w);

  if (input_.ndimension() == 3) {
    output.resize_({nplane, output_h, output_w});
  } else {
    output.resize_({nbatch, nplane, output_h, output_w});
  }
  if (output.numel() == 0) {
    return;
  }

  Tensor input = input_.contiguous();

  int64_t output_plane_size = output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);

  int64_t size_y = nplane;
  int64_t size_z = nbatch;

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16,
    input.scalar_type(), "reflection_pad2d_out_template", [&] {

      for (int64_t block_y = 0; block_y < size_y; block_y += 65535) {
        int64_t block_y_size = std::min(size_y - block_y, static_cast<int64_t>(65535));
        for (int64_t block_z = 0; block_z < size_z; block_z += 65535) {
          int64_t block_z_size = std::min(size_z - block_z, static_cast<int64_t>(65535));

          dim3 grid_size(at::ceil_div(output_plane_size, static_cast<int64_t>(256)), block_y_size, block_z_size);

          reflection_pad2d_out_kernel<<<
            grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
              input.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(),
              input_w, input_h,
              pad_t, pad_b, pad_l, pad_r, block_y, block_z, nplane);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      }
    }
  );
}

void reflection_pad2d_backward_out_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding) {
  if (grad_input.numel() == 0) {
    return;
  }

  int plane_dim = 0;
  int dim_h = 1;
  int dim_w = 2;
  int nbatch = 1;

  if (input.ndimension() == 4) {
    nbatch = input.size(0);
    plane_dim++;
    dim_h++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];
  int64_t pad_t = padding[2];
  int64_t pad_b = padding[3];

  int nplane = input.size(plane_dim);
  int input_h = input.size(dim_h);
  int input_w = input.size(dim_w);

  int output_h = input_h + pad_t + pad_b;
  int output_w = input_w + pad_l + pad_r;

  TORCH_CHECK(
      output_w == grad_output_.size(dim_w),
      "grad_output width "
      "unexpected. Expected: ",
      output_w,
      ", Got: ",
      grad_output_.size(dim_w));
  TORCH_CHECK(
      output_h == grad_output_.size(dim_h),
      "grad_output height "
      "unexpected. Expected: ",
      output_h,
      ", Got: ",
      grad_output_.size(dim_h));

  Tensor grad_output = grad_output_.contiguous();

  int64_t output_plane_size = output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);

  int64_t size_y = nplane;
  int64_t size_z = nbatch;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "reflection_pad2d_backward_out_template",
      [&] {
        if (at::globalContext().deterministicAlgorithms()) {
          const int grid_size = 1024;
          const int block_size = 256;

          reflection_pad2d_backward_det_out_kernel<<<
              grid_size,
              block_size,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              grad_input.mutable_data_ptr<scalar_t>(),
              grad_output.const_data_ptr<scalar_t>(),
              input_w,
              input_h,
              pad_t,
              pad_b,
              pad_l,
              pad_r,
              nbatch,
              nplane,
              0);
        } else {
          for (int64_t block_y = 0; block_y < size_y; block_y += 65535) {
            int64_t block_y_size =
                std::min(size_y - block_y, static_cast<int64_t>(65535));
            for (int64_t block_z = 0; block_z < size_z; block_z += 65535) {
              int64_t block_z_size =
                  std::min(size_z - block_z, static_cast<int64_t>(65535));

              dim3 grid_size(
                  at::ceil_div(output_plane_size, static_cast<int64_t>(256)),
                  block_y_size,
                  block_z_size);

              reflection_pad2d_backward_out_kernel<<<
                  grid_size,
                  block_size,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  grad_input.mutable_data_ptr<scalar_t>(),
                  grad_output.const_data_ptr<scalar_t>(),
                  input_w,
                  input_h,
                  pad_t,
                  pad_b,
                  pad_l,
                  pad_r,
                  block_y,
                  block_z,
                  nplane);

              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
          }
        }
      });
}

} // namespace

TORCH_IMPL_FUNC(reflection_pad1d_out_cuda)
(const Tensor& input_, IntArrayRef padding, const Tensor& output) {
  TORCH_CHECK(
      canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  if (output.numel() == 0) {
    return;
  }

  int64_t dim_plane = 0;
  int64_t dim_w = 1;
  int64_t nbatch = 1;

  if (input_.ndimension() == 3) {
    nbatch = input_.size(0);
    dim_plane++;
    dim_w++;
  }

  int64_t pad_l = padding[0];
  int64_t pad_r = padding[1];

  int64_t nplane = input_.size(dim_plane);
  int64_t input_w = input_.size(dim_w);
  int64_t output_w = input_w + pad_l + pad_r;

  dim3 block_size(output_w > 256 ? 256 : output_w);
  dim3 grid_size((int)::ceil(output_w / 256.0), nplane, nbatch);

  Tensor input = input_.contiguous();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input.scalar_type(), "reflection_pad1d_out_template", [&] {
        reflection_pad1d_out_kernel<<<
            grid_size,
            block_size,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            input.const_data_ptr<scalar_t>(),
            output.mutable_data_ptr<scalar_t>(),
            input_w,
            pad_l,
            pad_r);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_cuda)(const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding,
    const Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("reflection_pad1d_backward_out_cuda");
  grad_input.zero_();

  if (grad_input.numel() == 0) {
    return;
  }

  TORCH_CHECK(canUse32BitIndexMath(input),
    "input tensor must fit into 32-bit index math");

  TORCH_CHECK(canUse32BitIndexMath(grad_output_),
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

  dim3 block_size(output_w > 256 ? 256 : output_w);
  dim3 grid_size((int) ::ceil(output_w / 256.0), nplane, nbatch);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16,
    grad_input.scalar_type(), "reflection_pad1d_backward_out_cuda", [&] {
      reflection_pad1d_backward_out_kernel<<<
        grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
          grad_input.mutable_data_ptr<scalar_t>(), grad_output.const_data_ptr<scalar_t>(),
          input_w, pad_l, pad_r);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  );
}

Tensor& reflection_pad2d_out_cuda(const Tensor& input, IntArrayRef padding,
    Tensor& output) {
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor reflection_pad2d_cuda(const Tensor& input, IntArrayRef padding) {
  auto output = at::empty({0}, input.options());
  reflection_pad2d_out_template(output, input, padding);
  return output;
}

Tensor& reflection_pad2d_backward_out_cuda(const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor reflection_pad2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  reflection_pad2d_backward_out_template(
    grad_input, grad_output, input, padding);
  return grad_input;
}


TORCH_IMPL_FUNC(reflection_pad3d_out_cuda) (
  const Tensor& input_, IntArrayRef padding, const Tensor& output
  ) {
  TORCH_CHECK(
      canUse32BitIndexMath(input_),
      "input tensor must fit into 32-bit index math");

  if (output.numel() == 0) {
    return;
  }

  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  auto input = input_.contiguous();
  bool batch_mode = (input.dim() == 5);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16,
      input.scalar_type(), "reflection_pad3d_out_cuda", [&] {
        auto input_inner = input;
        auto output_inner = output;
        if (!batch_mode) {
          // non-batch mode
          input_inner = input.unsqueeze(0);
          output_inner = output.unsqueeze(0);
        }

        auto input_packed = input_inner.packed_accessor64<const scalar_t, 5>();
        auto output_packed = output_inner.packed_accessor64<scalar_t, 5>();

        int64_t output_plane_size = output_packed.size(2) * output_packed.size(3) * output_packed.size(4);
        int64_t size_y = input_packed.size(1);
        int64_t size_z = input_packed.size(0);
        dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);

        for (int64_t block_y = 0; block_y < size_y; block_y += 65535) {
          int64_t block_y_size = std::min(size_y - block_y, static_cast<int64_t>(65535));
          for (int64_t block_z = 0; block_z < size_z; block_z += 65535) {
            int64_t block_z_size = std::min(size_z - block_z, static_cast<int64_t>(65535));

            dim3 grid_size(at::ceil_div(output_plane_size, static_cast<int64_t>(256)), \
                           block_y_size, block_z_size);

            reflection_pad3d_out_kernel<<<
                grid_size, block_size,0, at::cuda::getCurrentCUDAStream()>>>(
                input_packed, output_packed, pad_left, pad_top, pad_front,
                block_y, block_z);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      });
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_cuda) (
  const Tensor& grad_output, const Tensor& input, IntArrayRef padding,
  const Tensor& grad_input) {
  globalContext().alertNotDeterministic("reflection_pad3d_backward_out_cuda");
  TORCH_CHECK(canUse32BitIndexMath(input), "input tensor must fit into 32-bit index math");
  TORCH_CHECK(canUse32BitIndexMath(grad_output), "input tensor must fit into 32-bit index math");

  if (grad_input.numel() == 0) {
    return;
  }
  grad_input.zero_();

  int64_t pad_left = padding[0];
  int64_t pad_top = padding[2];
  int64_t pad_front = padding[4];

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "reflection_pad3d_backward_out_cuda", [&] {
        auto grad_input_ = grad_input;
        auto grad_output_ = grad_output;
        if (input.dim() == 4) {
          // non-batch mode
          grad_input_ = grad_input.unsqueeze(0);
          grad_output_ = grad_output.unsqueeze(0);
        }

        auto grad_input_packed = grad_input_.packed_accessor64<scalar_t, 5>();
        auto grad_output_packed = grad_output_.packed_accessor64<const scalar_t, 5>();

        int64_t output_plane_size = grad_output_packed.size(2) *
            grad_output_packed.size(3) * grad_output_packed.size(4);
        int64_t size_y = grad_input_packed.size(1);
        int64_t size_z = grad_input_packed.size(0);
        dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);

        for (int64_t block_y = 0; block_y < size_y; block_y += 65535) {
          int64_t block_y_size = std::min(size_y - block_y, static_cast<int64_t>(65535));
          for (int64_t block_z = 0; block_z < size_z; block_z += 65535) {
            int64_t block_z_size = std::min(size_z - block_z, static_cast<int64_t>(65535));

            dim3 grid_size(at::ceil_div(output_plane_size, static_cast<int64_t>(256)), \
                           block_y_size, block_z_size);

            reflection_pad3d_backward_out_kernel<<<
                grid_size, block_size,0, at::cuda::getCurrentCUDAStream()>>>(
                grad_input_packed, grad_output_packed, pad_left, pad_top, pad_front,
                block_y, block_z);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        }
      });
}

} // namespace at::native
