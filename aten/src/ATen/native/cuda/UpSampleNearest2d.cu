#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/upsample_nearest2d_backward_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#endif

namespace at::native {
namespace {

#define MAX_THREADS 512

// Define a typedef to dispatch to nearest_neighbor_compute_source_index or
// nearest_neighbor_exact_compute_source_index
typedef int (*nn_compute_source_index_fn_t)(const float, int, int);

// Define a typedef to dispatch to nearest_neighbor_bw_compute_source_index or
// nearest_neighbor_exact_bw_compute_source_index
typedef int (*nn_bw_compute_source_index_fn_t)(const float, int, int);

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
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
      : nn_compute_source_index_fn(height_scale, h2, height1);
  const size_t w1 = width1 == width2
      ? w2
      : nn_compute_source_index_fn(width_scale, w2, width1);

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

template <typename scalar_t, nn_compute_source_index_fn_t nn_compute_source_index_fn>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest2d_nhwc_out_frame(
    const scalar_t* idata,
    scalar_t* odata,
    const size_t channels,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    float height_scale,
    float width_scale,
    const size_t out_numel) {

  const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < out_numel) {
    const auto c = index % channels;
    const auto w2 = (index / channels) % width2;
    const auto h2 = (index / channels / width2) % height2;
    const auto n = index / channels / width2 / height2;

    const size_t h1 = height1 == height2 ? h2 : nn_compute_source_index_fn(height_scale, h2, height1);
    const size_t w1 = width1 == width2 ? w2 : nn_compute_source_index_fn(width_scale, w2, width1);

    odata[index] = idata[idx_cl(n, h1, w1, c, height1, width1, channels)];
  }
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename scalar_t, typename accscalar_t, nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
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
      nn_bw_compute_source_index_fn(height_scale, dst_y, src_dim_h);
  int src_y_up = nn_bw_compute_source_index_fn(
      height_scale, dst_y + 1, src_dim_h);

  int dst_x = dst_idx % dst_dim_w;
  // note that we do not want to clamp src_x to src_dim_w, since we might
  // intentionally want to skip in case of scale_factor < 1.0
  int src_x =
      nn_bw_compute_source_index_fn(width_scale, dst_x, src_dim_w);
  int src_x_up = nn_bw_compute_source_index_fn(
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

template <typename scalar_t, typename accscalar_t, nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest2d_backward_nhwc_out_frame(
    const scalar_t* go,
    scalar_t* gi,
    const size_t height1,
    const size_t width1,
    const size_t height2,
    const size_t width2,
    const size_t channels,
    const float height_scale,
    const float width_scale,
    const size_t gi_numel) {

  // 1 is for grad_output (src)
  // 2 is for grad_input (dst)

  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < gi_numel) {
    const int c = index % channels;
    const int w2 = (index / channels) % width2;
    const int h2 = (index / channels / width2) % height2;
    const int n = index / channels / width2 / height2;

    int h1 = nn_bw_compute_source_index_fn(height_scale, h2, height1);
    int h1_up = nn_bw_compute_source_index_fn(height_scale, h2 + 1, height1);

    int w1 = nn_bw_compute_source_index_fn(width_scale, w2, width1);
    int w1_up = nn_bw_compute_source_index_fn(width_scale, w2 + 1, width1);

    accscalar_t grad = 0;
    for (int ih = h1; ih < h1_up; ih++) {
      for (int iw = w1; iw < w1_up; iw++) {
        grad += go[idx_cl(n, ih, iw, c, height1, width1, channels)];
      }
    }
    gi[index] = static_cast<scalar_t>(grad);
  }
}

template<nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest2d_out_cuda_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  if (input_.numel() == 0) {
    return;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_height = input_.size(2);
  int input_width = input_.size(3);

  const float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
  const float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

  const auto memory_format = input_.suggest_memory_format();

  if (input_.sizes() == output.sizes()) {
    output.copy_(input_);
    return;
  }

  // heuristic: only use channels_last path when it's faster than the contiguous path
  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 && \
        output.is_contiguous(memory_format)) {
    at::Tensor input = input_.contiguous(at::MemoryFormat::ChannelsLast);

    TORCH_CHECK(input.numel() < std::numeric_limits<int64_t>::max(),
      "upsample_nearest_nhwc only supports input tensors with less than 2^63 - 1 elements");
    TORCH_CHECK(output.numel() < std::numeric_limits<int64_t>::max(),
      "upsample_nearest_nhwc only supports output tensors with less than 2^63 - 1 elements");

    const int64_t num_kernels = output.numel();
    const int64_t num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, input.scalar_type(), "upsample_nearest2d_nhwc_out_frame", [&] {
      const scalar_t* idata = input.const_data_ptr<scalar_t>();
      scalar_t* odata = output.mutable_data_ptr<scalar_t>();

      upsample_nearest2d_nhwc_out_frame<scalar_t, nn_compute_source_index_fn>
        <<<ceil_div(num_kernels, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          idata,
          odata,
          channels,
          input_height,
          input_width,
          output_height,
          output_width,
          height_scale,
          width_scale,
          output.numel()
      );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
  else {
    // This is needed for non-contiguous tensors.
    Tensor output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
    Tensor input = input_.contiguous();

    int nc = nbatch * channels;

    const int max_threads = std::min<int>(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS);

    int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
    int* maxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;

    // upsample_nearest2d meta call makes sure input/output tensor is not empty;
    int block_x = std::min<int>(
        maxThreadsDim[0], std::min<int>(lastPow2(output_width), max_threads));
    int block_y = std::min<int>(
        maxThreadsDim[1],
        std::min<int>(lastPow2(output_height), max_threads / block_x));
    int block_z = std::min<int>(
        maxThreadsDim[2], std::min<int>(nc, max_threads / block_x / block_y));
    const dim3 block(block_x, block_y, block_z);

    int grid_x = ceil_div(output_width, block_x);
    int grid_y = ceil_div(output_height, block_y);
    int grid_z = std::min<int>(
        maxGridSize[2], ceil_div(nc, block_z * 4));
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
    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, input.scalar_type(), "upsample_nearest2d_out_frame", [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;

          auto idata = input.const_data_ptr<scalar_t>();
          auto odata = output_c.mutable_data_ptr<scalar_t>();

          upsample_nearest2d_out_frame<scalar_t, nn_compute_source_index_fn>
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

    if (!output.is_contiguous()) {
        output.copy_(output_c);
    }
  }
}

template<nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
static void upsample_nearest2d_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  if (grad_input.numel() == 0) {
    return;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  const float height_scale = compute_scales_value_backwards<float>(scales_h, output_height, input_height);
  const float width_scale = compute_scales_value_backwards<float>(scales_w, output_width, input_width);

  auto memory_format = grad_output_.suggest_memory_format();

  if (grad_output_.sizes() == grad_input.sizes()) {
    grad_input.copy_(grad_output_);
    return;
  }

  if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 && \
        grad_input.is_contiguous(memory_format)) {
    Tensor grad_output = grad_output_.contiguous(at::MemoryFormat::ChannelsLast);

    TORCH_CHECK(grad_input.numel() < std::numeric_limits<int>::max(),
      "upsample_nearest_nhwc only supports grad_input tensors with less than INT_MAX elements");
    TORCH_CHECK(grad_output.numel() < std::numeric_limits<int>::max(),
      "upsample_nearest_nhwc only supports grad_output tensors with less than INT_MAX elements");

    const int num_kernels = grad_input.numel();
    const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest2d_backward_nhwc_out_frame", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;

      const scalar_t* go = grad_output.const_data_ptr<scalar_t>();
      scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();

      upsample_nearest2d_backward_nhwc_out_frame<scalar_t, accscalar_t, nn_bw_compute_source_index_fn>
        <<<ceil_div(num_kernels, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          go,
          gi,
          output_height,
          output_width,
          input_height,
          input_width,
          channels,
          height_scale,
          width_scale,
          grad_input.numel()
      );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  } else {
    // This is needed for non-contiguous tensors.
    Tensor grad_input_c = grad_input.is_contiguous() ? grad_input : at::empty(grad_input.sizes(), grad_input.options());
    Tensor grad_output = grad_output_.contiguous();

    // upsample_nearest2d meta call makes sure `nbatch != 0`
    unsigned int n = grad_input.numel() / nbatch;
    dim3 bdim{std::min<unsigned int>(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
    dim3 gdim{ceil_div(n, bdim.x)};
    // safe check for int32 indexing; implicitly restrict launch config for kernel
    TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest2d_backward_out_frame", [&] {
      using accscalar_t = at::acc_type<scalar_t, true>;

      auto idata = grad_input_c.mutable_data_ptr<scalar_t>();
      auto odata = grad_output.const_data_ptr<scalar_t>();


      upsample_nearest2d_backward_out_frame<scalar_t, accscalar_t, nn_bw_compute_source_index_fn>
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

    if (!grad_input.is_contiguous()) {
        grad_input.copy_(grad_input_c);
    }
  }
}

} // namespace

TORCH_IMPL_FUNC(upsample_nearest2d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output) {
  upsample_nearest2d_out_cuda_template<nearest_neighbor_compute_source_index>(
      output, input, output_size, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output) {
  upsample_nearest2d_out_cuda_template<nearest_neighbor_exact_compute_source_index>(
      output, input, output_size, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  upsample_nearest2d_backward_out_cuda_template<nearest_neighbor_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  upsample_nearest2d_backward_out_cuda_template<nearest_neighbor_exact_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
}

} // namespace at::native
