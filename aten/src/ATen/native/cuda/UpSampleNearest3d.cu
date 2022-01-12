#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/UpSample.cuh>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/upsample_nearest3d.h>
#include <ATen/ops/upsample_nearest3d_native.h>
#include <ATen/ops/upsample_nearest3d_backward.h>
#include <ATen/ops/upsample_nearest3d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact3d.h>
#include <ATen/ops/_upsample_nearest_exact3d_native.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward_native.h>
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
    scalar_t* output,
    float depth_scale,
    float height_scale,
    float width_scale) {

  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w)
    return;

  int dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
  int src_c_stride = src_dim_d * src_dim_h * src_dim_w;

  int c = (dst_idx / (dst_c_stride)) % dim_c;

  int dst_z = (dst_idx / dst_dim_h / dst_dim_w) % dst_dim_d;
  int src_z = nn_compute_source_index_fn(depth_scale, dst_z, src_dim_d);
  int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
  int src_y = nn_compute_source_index_fn(height_scale, dst_y, src_dim_h);

  int dst_x = dst_idx % dst_dim_w;
  int src_x = nn_compute_source_index_fn(width_scale, dst_x, src_dim_w);

  int src_idx = c * src_c_stride + src_z * src_dim_h * src_dim_w +
      src_y * src_dim_w + src_x;
  for (int b = 0; b < dim_b; b++) {
    output[dst_idx] = input[src_idx];
    src_idx += dim_c * src_c_stride;
    dst_idx += dim_c * dst_c_stride;
  }
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
// Backward operation
template <typename scalar_t, typename accscalar_t, nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
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
    scalar_t* grad_i,
    float depth_scale,
    float height_scale,
    float width_scale) {

  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= dim_c * dst_dim_d * dst_dim_h * dst_dim_w)
    return;

  int dst_c_stride = dst_dim_d * dst_dim_h * dst_dim_w;
  int src_c_stride = src_dim_d * src_dim_h * src_dim_w;

  int c = (dst_idx / (dst_c_stride)) % dim_c;

  int dst_z = (dst_idx / dst_dim_h / dst_dim_w) % dst_dim_d;
  // note that we do not want to clamp src_z to src_dim_z, since we might
  // intentionally want to skip in case of scale_factor < 1.0
  int src_z = nn_bw_compute_source_index_fn(depth_scale, dst_z, src_dim_d);
  int src_z_up = nn_bw_compute_source_index_fn(depth_scale, dst_z+1, src_dim_d);

  int dst_y = (dst_idx / dst_dim_w) % dst_dim_h;
  // note that we do not want to clamp src_y to src_dim_y, since we might
  // intentionally want to skip in case of scale_factor < 1.0
  int src_y = nn_bw_compute_source_index_fn(height_scale, dst_y, src_dim_h);
  int src_y_up = nn_bw_compute_source_index_fn(height_scale, dst_y+1, src_dim_h);

  int dst_x = dst_idx % dst_dim_w;
  // note that we do not want to clamp src_x to src_dim_w, since we might
  // intentionally want to skip in case of scale_factor < 1.0
  int src_x = nn_bw_compute_source_index_fn(width_scale, dst_x, src_dim_w);
  int src_x_up = nn_bw_compute_source_index_fn(width_scale, dst_x+1, src_dim_w);

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

template<nn_compute_source_index_fn_t nn_compute_source_index_fn>
static void upsample_nearest3d_out_cuda_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  // TODO: remove this when the cuda kernel is updated to support the channels_last memory format.
  // This is a temporary hack to prevent a silence correctness issue when calling this kernel
  // with tensors in channels_last format.
  auto output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_.size(0);
  int channels = input_.size(1);
  int input_depth = input_.size(2);
  int input_height = input_.size(3);
  int input_width = input_.size(4);

  Tensor input = input_.contiguous();

  if (input.numel() == 0) {
    return;
  }

  // upsample_nearest3d meta call makes sure `nbatch != 0`
  unsigned int n = output.numel() / nbatch;
  dim3 bdim{std::min<unsigned int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
  dim3 gdim{ceil_div(n, bdim.x)};
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(output.numel() <= std::numeric_limits<int32_t>::max());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::Byte,input.scalar_type(), "upsample_nearest3d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.data_ptr<scalar_t>();
        auto odata = output_c.data_ptr<scalar_t>();

        const float depth_scale = compute_scales_value<float>(scales_d, input_depth, output_depth);
        const float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
        const float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

        upsample_nearest3d_out_frame<scalar_t, nn_compute_source_index_fn>
          <<<gdim, bdim, 0, stream>>>(
            idata,
            nbatch,
            channels,
            input_depth,
            input_height,
            input_width,
            output_depth,
            output_height,
            output_width,
            odata,
            depth_scale,
            height_scale,
            width_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  if (!output.is_contiguous()) {
      output.copy_(output_c);
  }
}

template<nn_bw_compute_source_index_fn_t nn_bw_compute_source_index_fn>
static void upsample_nearest3d_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      __func__,
      {grad_output_arg, grad_input_arg});

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_depth = input_size[2];
  int input_height = input_size[3];
  int input_width = input_size[4];

  Tensor grad_output = grad_output_.contiguous();

  if (grad_input.numel() == 0) {
    return;
  }

  // upsample_nearest3d meta call makes sure `nbatch != 0`
  unsigned int n = grad_input.numel() / nbatch;
  dim3 bdim{std::min<unsigned int>(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, MAX_THREADS)};
  dim3 gdim{ceil_div(n, bdim.x)};
  // safe check for int32 indexing; implicitly restrict launch config for kernel
  TORCH_CHECK(grad_input.numel() <= std::numeric_limits<int32_t>::max());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::Byte, grad_output.scalar_type(), "upsample_nearest3d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.data_ptr<scalar_t>();
        auto odata = grad_output.data_ptr<scalar_t>();

        float depth_scale = compute_scales_value_backwards<float>(scales_d, output_depth, input_depth);
        float height_scale = compute_scales_value_backwards<float>(scales_h, output_height, input_height);
        float width_scale = compute_scales_value_backwards<float>(scales_w, output_width, input_width);

        upsample_nearest3d_backward_out_frame<scalar_t, accscalar_t, nn_bw_compute_source_index_fn>
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
                idata,
                depth_scale,
                height_scale,
                width_scale);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_nearest3d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output) {
  upsample_nearest3d_out_cuda_template<nearest_neighbor_compute_source_index>(
      output, input, output_size, scales_d, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output) {
  upsample_nearest3d_out_cuda_template<nearest_neighbor_exact_compute_source_index>(output, input, output_size, scales_d, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_nearest3d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  upsample_nearest3d_backward_out_cuda_template<nearest_neighbor_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  upsample_nearest3d_backward_out_cuda_template<nearest_neighbor_exact_bw_compute_source_index>(
      grad_input, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}

using at::native::upsample::compute_output_size;
using at::native::upsample_cuda::get_scale_value;

Tensor upsample_nearest3d_cuda(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  return at::upsample_nearest3d(input, osize, scale_d, scale_h, scale_w);
}

Tensor _upsample_nearest_exact3d_cuda(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  return at::_upsample_nearest_exact3d(input, osize, scale_d, scale_h, scale_w);
}

// when structured kernels can handle QuantizedCPU, update these overloads to be CompositeExplicitAutograd
Tensor upsample_nearest3d_backward_cuda(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  return at::upsample_nearest3d_backward(grad_output, osize, input_size, scale_d, scale_h, scale_w);
}

Tensor _upsample_nearest_exact3d_backward_cuda(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  return at::_upsample_nearest_exact3d_backward(grad_output, osize, input_size, scale_d, scale_h, scale_w);
}

} // namespace native
} // namespace at
