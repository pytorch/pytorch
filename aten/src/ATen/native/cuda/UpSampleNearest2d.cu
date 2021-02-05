#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>

namespace at {
namespace native {
namespace {

#define MAX_THREADS 512

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest2d_out_frame(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t nc,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    float height_scale,
    float width_scale) {
  size_t nc_iter = threadIdx.z + blockIdx.z * blockDim.z;
  int w2 = threadIdx.x + blockIdx.x * blockDim.x;
  int h2 = threadIdx.y + blockIdx.y * blockDim.y;

  if (w2 >= width2 || h2 >= height2) {
    return;
  }

  int nc_stride = blockDim.z * gridDim.z;

  const size_t h1 = height1 == height2
      ? h2
      : nearest_neighbor_compute_source_index(height_scale, h2, height1);
  const size_t w1 = width1 == width2
      ? w2
      : nearest_neighbor_compute_source_index(width_scale, w2, width1);

  size_t src_index = (nc_iter * height1 + h1) * width1 + w1;
  size_t src_index_stride = nc_stride * width1 * height1;
  size_t dst_index = (nc_iter * height2 + h2) * width2 + w2;
  size_t dst_index_stride = nc_stride * width2 * height2;

  // iterating over
  while (nc_iter < nc) {
    odata[dst_index] = idata[src_index];
    dst_index += dst_index_stride;
    src_index += src_index_stride;
    nc_iter += nc_stride;
  }
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest2d_backward_out_frame(
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_h,
    size_t src_dim_w,
    size_t dst_dim_h,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float height_scale,
    float width_scale) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= dim_c * dst_dim_h * dst_dim_w)
    return;

  int dst_c_stride = dst_dim_h * dst_dim_w;
  int src_c_stride = src_dim_h * src_dim_w;

  int c = (dst_idx / (dst_c_stride)) % dim_c;

  int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
  // note that we do not want to clamp src_y to src_dim_y, since we might
  // intentionally want to skip in case of scale_factor < 1.0
  int src_y =
      nearest_neighbor_bw_compute_source_index(height_scale, dst_y, src_dim_h);
  int src_y_up = nearest_neighbor_bw_compute_source_index(
      height_scale, dst_y + 1, src_dim_h);

  int dst_x = dst_idx % dst_dim_w;
  // note that we do not want to clamp src_x to src_dim_w, since we might
  // intentionally want to skip in case of scale_factor < 1.0
  int src_x =
      nearest_neighbor_bw_compute_source_index(width_scale, dst_x, src_dim_w);
  int src_x_up = nearest_neighbor_bw_compute_source_index(
      width_scale, dst_x + 1, src_dim_w);

  for (int b = 0; b < dim_b; b++) {
    accscalar_t grad = 0;
    for (int y = src_y; y < src_y_up; y++) {
      for (int x = src_x; x < src_x_up; x++) {
        int src_idx =
            b * dim_c * src_c_stride + c * src_c_stride + y * src_dim_w + x;
        grad += grad_o[src_idx];
      }
    }
    grad_i[dst_idx] = grad;
    dst_idx += dim_c * dst_c_stride;
  }
}

static void upsample_nearest2d_out_cuda_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(
      "upsample_nearest2d_out_cuda_template", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_height = input_.size(2);
  int input_width = input_.size(3);

  upsample_2d_shape_check(
      input_,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  Tensor input = input_.contiguous();
  output.resize_({nbatch, channels, output_height, output_width});

  if (input.numel() == 0) {
    return;
  }

  int nc = nbatch * channels;

  const int max_threads = std::min<int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS);

  int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
  int* maxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;

  // upsample_2d_shape_check makes sure input/output tensor is not empty;
  int block_x = std::min<int>(
      maxThreadsDim[0], std::min<int>(lastPow2(output_width), max_threads));
  int block_y = std::min<int>(
      maxThreadsDim[1],
      std::min<int>(lastPow2(output_height), max_threads / block_x));
  int block_z = std::min<int>(
      maxThreadsDim[2], std::min<int>(nc, max_threads / block_x / block_y));
  const dim3 block(block_x, block_y, block_z);

  int grid_x = cuda::ATenCeilDiv(output_width, block_x);
  int grid_y = cuda::ATenCeilDiv(output_height, block_y);
  int grid_z = std::min<int>(
      maxGridSize[2], cuda::ATenCeilDiv(nc, block_z * 4));
  const dim3 grid(grid_x, grid_y, grid_z);
  // Error out on cases where grid_x & grid_y exceeds limit of launch config, as
  // the current kernel implementation doesn't loop over the two dimensions.
  // This is unlikely to happen.
  // TODO: kernel implementation could stride on spatial dimension. We probably
  //       need to overhaul the kernel.
  TORCH_CHECK(
      grid_x <= maxGridSize[0] && grid_y <= maxGridSize[1],
      "input tensor has spatial dimension larger than the kernel capacity");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::Byte, input.scalar_type(), "upsample_nearest2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.data_ptr<scalar_t>();
        auto odata = output.data_ptr<scalar_t>();

        const float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
        const float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

        upsample_nearest2d_out_frame<scalar_t, accscalar_t>
            <<<grid, block, 0, stream>>>(
                idata,
                odata,
                nc,
                input_height,
                input_width,
                output_height,
                output_width,
                height_scale,
                width_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

static void upsample_nearest2d_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_nearest2d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  upsample_2d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  Tensor grad_output = grad_output_.contiguous();
  grad_input.resize_({nbatch, channels, input_height, input_width});

  if (grad_input.numel() == 0) {
    return;
  }

  // upsample_2d_shape_check makes sure `nbatch != 0`
  unsigned int n = grad_input.numel() / nbatch;
  dim3 bdim{std::min<unsigned int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
  dim3 gdim{cuda::ATenCeilDiv(n, bdim.x)};
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest2d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.data_ptr<scalar_t>();
        auto odata = grad_output.data_ptr<scalar_t>();

        const float height_scale = compute_scales_value_backwards<float>(scales_h, output_height, input_height);
        const float width_scale = compute_scales_value_backwards<float>(scales_w, output_width, input_width);

        upsample_nearest2d_backward_out_frame<scalar_t, accscalar_t>
            <<<gdim, bdim, 0, stream>>>(
                odata,
                nbatch,
                channels,
                output_height,
                output_width,
                input_height,
                input_width,
                idata,
                height_scale,
                width_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_nearest2d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output) {
  upsample_nearest2d_out_cuda_template(output, input, output_size, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  upsample_nearest2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}

using at::native::upsample::compute_output_size;
using at::native::upsample_cuda::get_scale_value;

Tensor upsample_nearest2d_cuda(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  upsample_nearest2d_out_cuda_template(output, input, osize, scale_h, scale_w);
  return output;
}

Tensor upsample_nearest2d_backward_cuda(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  auto grad_input = at::empty_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  upsample_nearest2d_backward_out_cuda_template(
      grad_input, grad_output, osize, input_size, scale_h, scale_w);
  return grad_input;
}

} // namespace native
} // namespace at
