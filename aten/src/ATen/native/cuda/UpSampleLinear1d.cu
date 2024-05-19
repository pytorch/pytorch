// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/UpSample.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_linear1d_native.h>
#include <ATen/ops/upsample_linear1d_backward_native.h>
#endif

namespace at::native {
namespace {

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void upsample_linear1d_out_frame(
    const int n,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor64<const scalar_t, 3> idata,
    PackedTensorAccessor64<scalar_t, 3> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int width1 = idata.size(2);
  const int width2 = odata.size(2);

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][w1];
          odata[n][c][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val =
            w0lambda * idata[n][c][w1] + w1lambda * idata[n][c][w1 + w1p];
        odata[n][c][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void upsample_linear1d_out_frame_backward(
    const int n,
    const accscalar_t rwidth,
    const bool align_corners,
    PackedTensorAccessor64<scalar_t, 3> idata,
    const PackedTensorAccessor64<const scalar_t, 3> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int width1 = idata.size(2);
  const int width2 = odata.size(2);

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[n][c][w1];
          idata[n][c][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[n][c][w2];
        gpuAtomicAddNoReturn(&idata[n][c][w1], static_cast<scalar_t>(w0lambda * d2val));
        gpuAtomicAddNoReturn(
            &idata[n][c][w1 + w1p], static_cast<scalar_t>(w1lambda * d2val));
      }
    }
  }
}

static void upsample_linear1d_out_cuda_template(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_width = output_size[0];

  int input_width = input.size(2);

  output.zero_();

  AT_ASSERT(input_width > 0 && output_width > 0);

  const int num_kernels = output_width;
  const int num_threads = 512;
      //at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "upsample_linear1d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<const scalar_t, 3>();
        auto odata = output.packed_accessor64<scalar_t, 3>();

        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_width, output_width, align_corners, scales);

        upsample_linear1d_out_frame<scalar_t, accscalar_t>
            <<<ceil_div(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(num_kernels, rwidth, align_corners, idata, odata);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

static void upsample_linear1d_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales) {
  TensorArg grad_output_arg{grad_output_, "grad_output_", 1},
      grad_input_arg{grad_input, "grad_input", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  int output_width = output_size[0];

  int input_width = input_size[2];

  Tensor grad_output = grad_output_.contiguous();

  grad_input.zero_();

  const int num_kernels = output_width;
  const int num_threads = 512;
      //at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad_output.scalar_type(), "upsample_linear1d_out_frame_backward", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor64<scalar_t, 3>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 3>();

        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales);

        upsample_linear1d_out_frame_backward<scalar_t, accscalar_t>
            <<<ceil_div(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(num_kernels, rwidth, align_corners, idata, odata);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_linear1d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales,
    const Tensor& output
) {
  upsample_linear1d_out_cuda_template(output, input, output_size, align_corners, scales);
}

TORCH_IMPL_FUNC(upsample_linear1d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales,
    const Tensor& grad_input
) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_linear1d_backward_out_cuda");
  upsample_linear1d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales);
}

} // namespace at::native
