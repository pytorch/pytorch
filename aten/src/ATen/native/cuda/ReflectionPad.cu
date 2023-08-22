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
template <typename scalar_t, typename F>
__device__ inline void parallel_reflection_pad3d(
    PackedTensorAccessor64<scalar_t, 5> input,
    PackedTensorAccessor64<scalar_t, 5> output,
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
    PackedTensorAccessor64<scalar_t, 5> input,
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
    PackedTensorAccessor64<scalar_t, 5> grad_output,
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
    Tensor &grad_input, const Tensor &grad_output_,
    const Tensor &input, IntArrayRef padding) {

  if (grad_input.numel() == 0) {
    return;
  }

  TORCH_CHECK(canUse32BitIndexMath(input),
    "input tensor must fit into 32-bit index math");
  TORCH_CHECK(canUse32BitIndexMath(grad_output_),
    "output gradient tensor must fit into 32-bit index math");

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
  int output_w  = input_w + pad_l + pad_r;

  TORCH_CHECK(output_w == grad_output_.size(dim_w), "grad_output width "
    "unexpected. Expected: ", output_w, ", Got: ", grad_output_.size(dim_w));
  TORCH_CHECK(output_h == grad_output_.size(dim_h), "grad_output height "
    "unexpected. Expected: ", output_h, ", Got: ", grad_output_.size(dim_h));

  Tensor grad_output = grad_output_.contiguous();

  int64_t output_plane_size = output_h * output_w;
  dim3 block_size(output_plane_size > 256 ? 256 : output_plane_size);

  int64_t size_y = nplane;
  int64_t size_z = nbatch;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16,
    input.scalar_type(), "reflection_pad2d_backward_out_template", [&] {

      for (int64_t block_y = 0; block_y < size_y; block_y += 65535) {
        int64_t block_y_size = std::min(size_y - block_y, static_cast<int64_t>(65535));
        for (int64_t block_z = 0; block_z < size_z; block_z += 65535) {
          int64_t block_z_size = std::min(size_z - block_z, static_cast<int64_t>(65535));

          dim3 grid_size(at::ceil_div(output_plane_size, static_cast<int64_t>(256)), block_y_size, block_z_size);

          reflection_pad2d_backward_out_kernel<<<
            grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_input.mutable_data_ptr<scalar_t>(), grad_output.const_data_ptr<scalar_t>(),
              input_w, input_h,
              pad_t, pad_b, pad_l, pad_r, block_y, block_z, nplane);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      }
    }
  );
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
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("reflection_pad2d_backward_out_cuda");
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
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("reflection_pad2d_backward_cuda");
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

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16,
      input.scalar_type(), "reflection_pad3d_out_cuda", [&] {
        auto input_inner = input;
        auto output_inner = output;
        if (!batch_mode) {
          // non-batch mode
          input_inner = input.unsqueeze(0);
          output_inner = output.unsqueeze(0);
        }

        auto input_packed = input_inner.packed_accessor64<scalar_t, 5>();
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
        auto grad_output_packed = grad_output_.packed_accessor64<scalar_t, 5>();

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
