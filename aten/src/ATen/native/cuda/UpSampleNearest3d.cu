#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/LegacyTHFunctionsCUDA.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>

namespace at {
namespace native {
namespace {

#define MAX_THREADS 512

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest3d_out_frame(
    const scalar_t* input,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_d,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_d,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* output) {

  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w)
    return;

  int dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
  int src_c_stride = src_dim_d * src_dim_h * src_dim_w;

  int c = (dst_idx / (dst_c_stride)) % dim_c;

  float scale_factor = (float)src_dim_d / (float)dst_dim_d;
  int dst_z = (dst_idx / dst_dim_h / dst_dim_w) % dst_dim_d;
  int src_z = nearest_neighbor_compute_source_index(scale_factor, dst_z, src_dim_d);

  scale_factor = (float)src_dim_h / (float)dst_dim_h;
  int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
  int src_y = nearest_neighbor_compute_source_index(scale_factor, dst_y, src_dim_h);

  scale_factor = (float)src_dim_w / (float)dst_dim_w;
  int dst_x = dst_idx % dst_dim_w;
  int src_x = nearest_neighbor_compute_source_index(scale_factor, dst_x, src_dim_w);

  int src_idx = c * src_c_stride + src_z * src_dim_h * src_dim_w +
      src_y * src_dim_w + src_x;
  for (int b = 0; b < dim_b; b++) {
    output[dst_idx] = input[src_idx];
    src_idx += dim_c * src_c_stride;
    dst_idx += dim_c * dst_c_stride;
  }
}

// Backward operation
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest3d_backward_out_frame(
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_d,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_d,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* grad_i) {

  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w)
    return;

  int dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
  int src_c_stride = src_dim_d * src_dim_h * src_dim_w;

  int c = (dst_idx / (dst_c_stride)) % dim_c;

  float scale_factor = (float)src_dim_d / (float)dst_dim_d;
  int dst_z = (dst_idx / dst_dim_h / dst_dim_w) % dst_dim_d;
  int src_z = nearest_neighbor_compute_source_index(scale_factor, dst_z, src_dim_d);
  int src_z_up = nearest_neighbor_compute_source_index(scale_factor, dst_z+1, src_dim_d+1);

  scale_factor = (float)src_dim_h / (float)dst_dim_h;
  int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
  int src_y = nearest_neighbor_compute_source_index(scale_factor, dst_y, src_dim_h);
  int src_y_up = nearest_neighbor_compute_source_index(scale_factor, dst_y+1, src_dim_h+1);

  scale_factor = (float)src_dim_w / (float)dst_dim_w;
  int dst_x = dst_idx % dst_dim_w;
  int src_x = nearest_neighbor_compute_source_index(scale_factor, dst_x, src_dim_w);
  int src_x_up = nearest_neighbor_compute_source_index(scale_factor, dst_x+1, src_dim_w+1);

  for (int b = 0; b < dim_b; b++) {
    accscalar_t grad = 0;
    for (int z = src_z; z < src_z_up; z++) {
      for (int y = src_y; y < src_y_up; y++) {
        for (int x = src_x; x < src_x_up; x++) {
          int src_idx = b * dim_c * src_c_stride + c * src_c_stride +
              z * src_dim_h * src_dim_w + y * src_dim_w + x;
          grad += grad_o[src_idx];
        }
      }
    }
    grad_i[dst_idx] = grad;
    dst_idx += dim_c * dst_c_stride;
  }
}

static void upsample_nearest3d_out_cuda_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_nearest3d_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_depth = input_.size(2);
  int input_height = input_.size(3);
  int input_width = input_.size(4);

  upsample_3d_shape_check(
      input_,
      Tensor(),
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);

  AT_ASSERT(
      input_depth > 0 && input_height > 0 && input_width > 0 &&
      output_depth > 0 && output_height > 0 && output_width > 0);

  Tensor input = input_.contiguous();
  output.resize_({input.size(0),
                  input.size(1),
                  output_depth,
                  output_height,
                  output_width});

  // upsample_3d_shape_check makes sure `nbatch != 0`
  unsigned int n = output.numel() / nbatch;
  dim3 bdim{std::min<unsigned int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
  dim3 gdim{cuda::ATenCeilDiv(n, bdim.x)};
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_nearest3d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.data_ptr<scalar_t>();
        auto odata = output.data_ptr<scalar_t>();

        upsample_nearest3d_out_frame<scalar_t><<<gdim, bdim, 0, stream>>>(
            idata,
            nbatch,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            odata);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

static void upsample_nearest3d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_nearest3d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  upsample_3d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_depth,
      input_height,
      input_width,
      output_depth,
      output_height,
      output_width);

  Tensor grad_output = grad_output_.contiguous();
  grad_input.resize_(
      {nbatch, channels, input_depth, input_height, input_width});

  // upsample_3d_shape_check makes sure `nbatch != 0`
  unsigned int n = grad_input.numel() / nbatch;
  dim3 bdim{std::min<unsigned int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
  dim3 gdim{cuda::ATenCeilDiv(n, bdim.x)};
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_nearest3d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.data_ptr<scalar_t>();
        auto odata = grad_output.data_ptr<scalar_t>();

        upsample_nearest3d_backward_out_frame<scalar_t, accscalar_t>
            <<<gdim, bdim, 0, stream>>>(
                odata,
                nbatch,
                channels,
                output_depth,
                output_height,
                output_width,
                input_depth,
                input_height,
                input_width,
                idata);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

Tensor& upsample_nearest3d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  upsample_nearest3d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor upsample_nearest3d_cuda(const Tensor& input, IntArrayRef output_size) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  upsample_nearest3d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor& upsample_nearest3d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  upsample_nearest3d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

Tensor upsample_nearest3d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  Tensor grad_input = at::empty_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  upsample_nearest3d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

} // namespace native
} // namespace at
