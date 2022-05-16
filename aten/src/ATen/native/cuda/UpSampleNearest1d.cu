#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/UpSample.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_nearest1d_native.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#endif

namespace at {
namespace native {
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
__global__ void upsample_nearest1d_out_frame(
    const scalar_t* input,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* output,
    float scale_factor) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= dim_c * dst_dim_w)
    return;

  int c = (dst_idx / dst_dim_w) % dim_c;

  int dst_x = dst_idx % dst_dim_w;
  int src_x = nn_compute_source_index_fn(scale_factor, dst_x, src_dim_w);

  int src_idx = c * src_dim_w + src_x;
  int src_stride = dim_c * src_dim_w;
  int dst_stride = dim_c * dst_dim_w;

  for (int b = 0; b < dim_b; b++) {
    output[dst_idx] = input[src_idx];
    src_idx += src_stride;
    dst_idx += dst_stride;
  }
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
// Backward operation
template <typename scalar_t, typename accscalar_t, nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest1d_backward_out_frame(
    const scalar_t* grad_o,
    size_t dim_b,
    size_t dim_c,
    size_t src_dim_w,
    size_t dst_dim_w,
    scalar_t* grad_i,
    float scale_factor) {

  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= dim_c * dst_dim_w)
    return;

  int c = (dst_idx / (dst_dim_w)) % dim_c;

  int dst_x = dst_idx % dst_dim_w;
  // note that we do not want to clamp src_x to src_dim_w, since we might
  // intentionally want to skip in case of scale_factor < 1.0
  int src_x = nn_bw_compute_source_index_fn(scale_factor, dst_x, src_dim_w);
  int src_x_up = nn_bw_compute_source_index_fn(scale_factor, dst_x+1, src_dim_w);

  for (int b = 0; b < dim_b; b++) {
    accscalar_t grad = 0;
    int src_idx = b * dim_c * src_dim_w + c * src_dim_w + src_x;
    for (int x = src_x; x < src_x_up; x++) {
      grad += grad_o[src_idx++];
    }
    grad_i[dst_idx] = grad;
    dst_idx += dim_c * dst_dim_w;
  }
}

template<nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest1d_out_cuda_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_nearest1d_out_cuda", {input_arg, output_arg});

  int output_width = output_size[0];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_width = input_.size(2);

  Tensor input = input_.contiguous();

  if (input.numel() == 0) {
    return;
  }

  // upsample_nearest1d meta call makes sure `nbatch != 0`
  unsigned int n = output.numel() / nbatch;
  dim3 bdim{std::min<unsigned int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
  dim3 gdim{ceil_div(n, bdim.x)};
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::Byte, input.scalar_type(), "upsample_nearest1d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.data_ptr<scalar_t>();
        auto odata = output.data_ptr<scalar_t>();

        const float scale_factor = compute_scales_value<float>(scales, input_width, output_width);

        upsample_nearest1d_out_frame<scalar_t, nn_compute_source_index_fn><<<gdim, bdim, 0, stream>>>(
            idata, nbatch, channels, input_width, output_width, odata, scale_factor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

template<nn_compute_source_index_fn_t nn_bw_compute_source_index_fn>
static void upsample_nearest1d_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_nearest1d_backward_out_cuda_template",
      {grad_output_arg, grad_input_arg});

  int output_width = output_size[0];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_width = input_size[2];

  Tensor grad_output = grad_output_.contiguous();

  if (grad_input.numel() == 0) {
    return;
  }

  // upsample_nearest1d meta call makes sure `nbatch != 0`
  unsigned int n = grad_input.numel() / nbatch;
  dim3 bdim{std::min<unsigned int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
  dim3 gdim{ceil_div(n, bdim.x)};
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest1d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.data_ptr<scalar_t>();
        auto odata = grad_output.data_ptr<scalar_t>();

        const float scale_factor = compute_scales_value_backwards<float>(scales, output_width, input_width);

        upsample_nearest1d_backward_out_frame<scalar_t, accscalar_t, nn_bw_compute_source_index_fn>
            <<<gdim, bdim, 0, stream>>>(
                odata, nbatch, channels, output_width, input_width, idata, scale_factor);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_nearest1d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    const Tensor& output
) {
  upsample_nearest1d_out_cuda_template<nearest_neighbor_compute_source_index>(
      output, input, output_size, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    const Tensor& output
) {
  upsample_nearest1d_out_cuda_template<nearest_neighbor_exact_compute_source_index>(output, input, output_size, scales);
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales,
    const Tensor& grad_input
) {
  upsample_nearest1d_backward_out_cuda_template<nearest_neighbor_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales,
    const Tensor& grad_input
) {
  upsample_nearest1d_backward_out_cuda_template<nearest_neighbor_exact_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales);
}

} // namespace native
} // namespace at
