// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/LaunchUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_bicubic2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bicubic2d_aa_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {
namespace {

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_out_frame(
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<const scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int height1 = idata.size(2);
  const int width1 = idata.size(3);
  const int width2 = odata.size(3);

  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1

    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
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
        const accscalar_t val = h0lambda *
                (w0lambda * idata[n][c][h1][w1] +
                 w1lambda * idata[n][c][h1][w1 + w1p]) +
            h1lambda *
                (w0lambda * idata[n][c][h1 + h1p][w1] +
                 w1lambda * idata[n][c][h1 + h1p][w1 + w1p]);
        odata[n][c][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_nhwc_out_frame(
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const int channels,
    const int height1,
    const int width1,
    const int height2,
    const int width2,
    const scalar_t* idata,
    scalar_t* odata,
    const int out_numel) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < out_numel) {
    const int c = index % channels;
    const int w2 = (index / channels) % width2;
    const int h2 = (index / channels / width2) % height2;
    const int n = index / channels / width2 / height2;

    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    const accscalar_t val = h0lambda * (
        w0lambda * idata[idx_cl(n, h1, w1, c, height1, width1, channels)] +
        w1lambda * idata[idx_cl(n, h1, w1 + w1p, c, height1, width1, channels)]
      ) + h1lambda * (
        w0lambda * idata[idx_cl(n, h1 + h1p, w1, c, height1, width1, channels)] +
        w1lambda * idata[idx_cl(n, h1 + h1p, w1 + w1p, c, height1, width1, channels)]
      );
    odata[idx_cl(n, h2, w2, c, height2, width2, channels)] = static_cast<scalar_t>(val);
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_backward_out_frame(
    const size_t nc,
    const int height1,
    const int width1,
    const int height2,
    const int width2,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* __restrict__ idata,
    const scalar_t* __restrict__ odata) {
  const size_t o_numel = nc * width2 * height2;
  const size_t i_numel = nc * width1 * height1;
  for (size_t index = blockDim.x * blockIdx.x + threadIdx.x; index < o_numel;
       index += blockDim.x * gridDim.x) {
    size_t index_temp = index;
    const int w2 = index_temp % width2; // 0:width2-1
    index_temp /= width2;
    const int h2 = index_temp % height2; // 0:height2-1
    const size_t nc = index_temp / height2;
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    const scalar_t d2val = odata[index];
    fastAtomicAdd(
        idata,
        idx(nc, height1, width1, h1, w1),
        i_numel,
        static_cast<scalar_t>(h0lambda * w0lambda * d2val),
        true);
    fastAtomicAdd(
        idata,
        idx(nc, height1, width1, h1, w1 + w1p),
        i_numel,
        static_cast<scalar_t>(h0lambda * w1lambda * d2val),
        true);
    fastAtomicAdd(
        idata,
        idx(nc, height1, width1, h1 + h1p, w1),
        i_numel,
        static_cast<scalar_t>(h1lambda * w0lambda * d2val),
        true);
    fastAtomicAdd(
        idata,
        idx(nc, height1, width1, h1 + h1p, w1 + w1p),
        i_numel,
        static_cast<scalar_t>(h1lambda * w1lambda * d2val),
        true);
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_backward_nhwc_out_frame(
    const int height1,
    const int width1,
    const int height2,
    const int width2,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* __restrict__ idata,
    const scalar_t* __restrict__ odata,
    const int channels,
    const size_t o_numel,
    const size_t i_numel) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < o_numel) {
    const int c = index % channels;
    const int w2 = (index / channels) % width2;
    const int h2 = (index / channels / width2) % height2;
    const int n = index / channels / width2 / height2;

    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

    const scalar_t d2val = odata[index];
    fastAtomicAdd(
        idata,
        idx_cl(n, h1, w1, c, height1, width1, channels),
        i_numel,
        static_cast<scalar_t>(h0lambda * w0lambda * d2val),
        true);
    fastAtomicAdd(
        idata,
        idx_cl(n, h1, w1 + w1p, c, height1, width1, channels),
        i_numel,
        static_cast<scalar_t>(h0lambda * w1lambda * d2val),
        true);
    fastAtomicAdd(
        idata,
        idx_cl(n, h1 + h1p, w1, c, height1, width1, channels),
        i_numel,
        static_cast<scalar_t>(h1lambda * w0lambda * d2val),
        true);
    fastAtomicAdd(
        idata,
        idx_cl(n, h1 + h1p, w1 + w1p, c, height1, width1, channels),
        i_numel,
        static_cast<scalar_t>(h1lambda * w1lambda * d2val),
        true);
  }
}

static void upsample_bilinear2d_out_cuda_template(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  const auto memory_format = input.suggest_memory_format();

  if (input.sizes() == output.sizes()) {
    output.copy_(input);
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
    // heuristic: only use channels_last path when it's faster than the contiguous path
    if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 16 && \
          output.is_contiguous(memory_format)) {
      using accscalar_t = at::acc_type<scalar_t, true>;

      TORCH_CHECK(input.numel() < std::numeric_limits<int>::max(),
        "upsample_bilinear2d_nhwc only supports input tensors with less than INT_MAX elements");
      TORCH_CHECK(output.numel() < std::numeric_limits<int>::max(),
        "upsample_bilinear2d_nhwc only supports output tensors with less than INT_MAX elements");

      const int channels = input.size(1);
      const int height1 = input.size(2);
      const int width1 = input.size(3);
      const int height2 = output.size(2);
      const int width2 = output.size(3);

      // const int num_kernels = output_height * output_width;
      const int num_kernels = output.numel();
      const int num_threads = std::min(
          at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

      at::Tensor input_cl = input.contiguous(at::MemoryFormat::ChannelsLast);

      const scalar_t* idata = input_cl.const_data_ptr<scalar_t>();
      scalar_t* odata = output.mutable_data_ptr<scalar_t>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_height, output_height, align_corners, scales_h);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_width, output_width, align_corners, scales_w);

      upsample_bilinear2d_nhwc_out_frame<scalar_t, accscalar_t>
        <<<ceil_div(num_kernels, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          rheight, rwidth, align_corners,
          channels,
          height1,
          width1,
          height2,
          width2,
          idata, odata,
          output.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // non-channels_last case, not necessarily contiguous
      const int num_kernels = output_height * output_width;
      const int num_threads = std::min(
          at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();

      using accscalar_t = at::acc_type<scalar_t, true>;

      auto idata = input.packed_accessor64<const scalar_t, 4>();
      auto odata = output.packed_accessor64<scalar_t, 4>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_height, output_height, align_corners, scales_h);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_width, output_width, align_corners, scales_w);

      upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
          <<<ceil_div(num_kernels, num_threads),
             num_threads,
             0,
             stream>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });
}

static void upsample_bilinear2d_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  if (grad_input.numel() == 0) {
    return;
  }

  const auto memory_format = grad_output_.suggest_memory_format();

  // initialization to zero is required here. As we launch one thread per output
  // element, and atomicAdd to input gradient. Given a sparse sampling case, our
  // threads are not covering the whole input tensor.
  grad_input.zero_();

  const size_t num_kernels = nbatch * channels * output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (grad_output_.sizes() == grad_input.sizes()) {
    grad_input.copy_(grad_output_);
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad_output_.scalar_type(), "upsample_bilinear2d_backward_out_frame", [&] {
    if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 && \
          grad_input.is_contiguous(memory_format)) {
      using accscalar_t = at::acc_type<scalar_t, true>;

      Tensor grad_output = grad_output_.contiguous(at::MemoryFormat::ChannelsLast);

      auto idata = grad_input.mutable_data_ptr<scalar_t>();
      auto odata = grad_output.const_data_ptr<scalar_t>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_height, output_height, align_corners, scales_h);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_width, output_width, align_corners, scales_w);

      upsample_bilinear2d_backward_nhwc_out_frame<scalar_t, accscalar_t>
          <<<ceil_div(num_kernels, static_cast<size_t>(num_threads)), num_threads, 0, stream>>>(
              input_height,
              input_width,
              output_height,
              output_width,
              rheight,
              rwidth,
              align_corners,
              idata,
              odata,
              channels,
              grad_output.numel(),
              grad_input.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      using accscalar_t = at::acc_type<scalar_t, true>;

      // This is needed for non-contiguous tensors.
      Tensor grad_input_c = grad_input.is_contiguous() ? grad_input : at::zeros(grad_input.sizes(), grad_input.options());
      Tensor grad_output = grad_output_.contiguous();

      auto idata = grad_input_c.mutable_data_ptr<scalar_t>();
      auto odata = grad_output.const_data_ptr<scalar_t>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_height, output_height, align_corners, scales_h);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_width, output_width, align_corners, scales_w);

      upsample_bilinear2d_backward_out_frame<scalar_t, accscalar_t>
          <<<ceil_div(num_kernels, static_cast<size_t>(num_threads)),
             num_threads,
             0,
             stream>>>(
              nbatch * channels,
              input_height,
              input_width,
              output_height,
              output_width,
              rheight,
              rwidth,
              align_corners,
              idata,
              odata);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      if (!grad_input.is_contiguous()) {
          grad_input.copy_(grad_input_c);
      }
    }
  });
}

// Code for upsampling with antialias
template <typename scalar_t, typename accscalar_t, typename InterpFilter>
C10_LAUNCH_BOUNDS_1(256) // 256 performs better then 1024
__global__ void upsample_gen2d_aa_out_frame(
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const PackedTensorAccessor64<const scalar_t, 4> idata,
    PackedTensorAccessor64<scalar_t, 4> odata,
    const InterpFilter & interp_filter) {

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int input_height = idata.size(2);
  const int input_width = idata.size(3);
  const int output_height = odata.size(2);
  const int output_width = odata.size(3);

  const int output_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int output_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (output_x >= output_width || output_y >= output_height) {
    return;
  }

  const accscalar_t half = 0.5;
  const accscalar_t support_h = static_cast<accscalar_t>(
      (height_scale >= 1.0) ? (interp_filter.size * half) * height_scale : interp_filter.size * half);
  const accscalar_t support_w = static_cast<accscalar_t>(
      (width_scale >= 1.0) ? (interp_filter.size * half) * width_scale : interp_filter.size * half);

  const int interp_height = (int)ceilf(support_h) * 2 + 1;
  const int interp_width = (int)ceilf(support_w) * 2 + 1;

  // Setup weights and a buffer using shared memory
  extern __shared__ int smem[];
  scalar_t* wx = reinterpret_cast<scalar_t*>(smem) + interp_width * threadIdx.x;
  scalar_t* wy = reinterpret_cast<scalar_t*>(smem) + interp_width * blockDim.x + interp_height * threadIdx.y;
  const int offset = interp_width * blockDim.x + interp_height * blockDim.y;
  scalar_t *buffer2 = reinterpret_cast<scalar_t*>(smem) + offset + \
      interp_height * (threadIdx.x + threadIdx.y * blockDim.x);

  // Compute weights and kernel spans
  int xmin, xsize, ymin, ysize;
  accscalar_t xcenter, ycenter;
  upsample_antialias::_compute_weights_span(
      output_x, input_width, width_scale, support_w, xmin, xsize, xcenter);
  upsample_antialias::_compute_weights_span(
      output_y, input_height, height_scale, support_h, ymin, ysize, ycenter);

  if (threadIdx.y == 0)
  {
    // All threadIdx.y have the same wx weights
    upsample_antialias::_compute_weights<scalar_t, accscalar_t>(
        wx,
        width_scale,
        interp_width,
        interp_filter,
        xmin - xcenter,
        xsize);
  }

  if (threadIdx.x == 0)
  {
    // All threadIdx.x have the same wy weights
    upsample_antialias::_compute_weights<scalar_t, accscalar_t>(
        wy,
        height_scale,
        interp_height,
        interp_filter,
        ymin - ycenter,
        ysize);
  }

  __syncthreads();

  const scalar_t * buffer1;

  // Parallelized across batch/channels
  for (int i = blockIdx.z; i < batchsize * channels; i += gridDim.z) {
    int n = i / channels;
    int c = i % channels;
    // interpolate on y-axis for ymin to ymin + ysize
    for (int y = 0; y < ysize; y++) {
      buffer1 = &(idata[n][c][ymin + y][xmin]);
      buffer2[y] = static_cast<scalar_t>(
          upsample_antialias::interpolate_aa_single_dim<scalar_t, accscalar_t>(
              buffer1, wx, xsize));
    }
    odata[n][c][output_y][output_x] = static_cast<scalar_t>(
        upsample_antialias::interpolate_aa_single_dim<scalar_t, accscalar_t>(
            buffer2, wy, ysize));
  }
}

// Code for upsampling with antialias
template <typename scalar_t, typename accscalar_t, typename InterpFilter>
C10_LAUNCH_BOUNDS_1(256) // 256 performs better then 1024
__global__ void upsample_gen2d_aa_backward_out_frame(
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    PackedTensorAccessor64<scalar_t, 4> idata,
    const PackedTensorAccessor64<const scalar_t, 4> odata,
    const InterpFilter & interp_filter) {

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int input_height = idata.size(2);
  const int input_width = idata.size(3);
  const int output_height = odata.size(2);
  const int output_width = odata.size(3);

  const int output_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int output_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (output_x >= output_width || output_y >= output_height) {
    return;
  }

  // special case: output just copy
  if (input_height == output_height && input_width == output_width) {
    for (int i = blockIdx.z; i < batchsize * channels; i += gridDim.z) {
      int n = i / channels;
      int c = i % channels;
      const scalar_t val = odata[n][c][output_y][output_x];
      idata[n][c][output_y][output_x] = val;
    }
    return;
  }

  const accscalar_t support_h = static_cast<accscalar_t>(
      (height_scale >= 1.0) ? (interp_filter.size * 0.5) * height_scale
                            : interp_filter.size * 0.5);
  const accscalar_t support_w = static_cast<accscalar_t>(
      (width_scale >= 1.0) ? (interp_filter.size * 0.5) * width_scale
                           : interp_filter.size * 0.5);

  const int interp_height = (int)ceilf(support_h) * 2 + 1;
  const int interp_width = (int)ceilf(support_w) * 2 + 1;

  // Setup weights using shared memory
  extern __shared__ int smem[];
  scalar_t* wx = reinterpret_cast<scalar_t*>(smem) + interp_width * threadIdx.x;
  scalar_t* wy = reinterpret_cast<scalar_t*>(smem) + interp_width * blockDim.x + interp_height * threadIdx.y;

  // Compute weights and kernel spans
  int xmin, xsize, ymin, ysize;
  accscalar_t xcenter, ycenter;
  upsample_antialias::_compute_weights_span(
      output_x, input_width, width_scale, support_w, xmin, xsize, xcenter);
  upsample_antialias::_compute_weights_span(
      output_y, input_height, height_scale, support_h, ymin, ysize, ycenter);

  if (threadIdx.y == 0)
  {
    // All threadIdx.y have the same wx weights
    upsample_antialias::_compute_weights<scalar_t, accscalar_t>(
        wx,
        width_scale,
        interp_width,
        interp_filter,
        xmin - xcenter,
        xsize);
  }

  if (threadIdx.x == 0)
  {
    // All threadIdx.x have the same wy weights
    upsample_antialias::_compute_weights<scalar_t, accscalar_t>(
        wy,
        height_scale,
        interp_height,
        interp_filter,
        ymin - ycenter,
        ysize);
  }

  __syncthreads();

  // Parallelized across batch/channels
  for (int i = blockIdx.z; i < batchsize * channels; i += gridDim.z) {
    int n = i / channels;
    int c = i % channels;
    scalar_t out_value = odata[n][c][output_y][output_x];
    for (int y = 0; y < ysize; y++) {
      for (int x = 0; x < xsize; x++) {
        upsample_increment_value_bounded<scalar_t, accscalar_t>(
            idata,
            n,
            c,
            input_height,
            input_width,
            ymin + y,
            xmin + x,
            wx[x] * wy[y] * out_value);
      }
    }
  }
}

// In the code below interp_filter_t distinguishes between bilinear and bicubic interpolations
// InterpFilter as BilinearFilterFunctor <--> bilinear
// InterpFilter as BicubicFilterFunctor <--> bicubic
template<typename InterpFilter>
static void upsample_gen2d_aa_out_cuda_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_gen2d_aa_out_cuda", {input_arg, output_arg});

  // TODO: remove this when the cuda kernel is updated to support the channels_last memory format.
  // This is a temporary hack to prevent a silence correctness issue when calling this kernel
  // with tensors in channels_last format.
  auto output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
  auto input = input_.contiguous();

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  size_t sharedMemPerBlock = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
  int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
  int maxThreadsPerBlock = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
  int* maxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;
  int block_x = std::min<int>(maxThreadsDim[0], at::cuda::warp_size());
  int grid_x = std::min<int>(maxGridSize[0], ceil_div(output_width, block_x));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<const scalar_t, 4>();
        auto odata = output_c.packed_accessor64<scalar_t, 4>();

        const accscalar_t height_scale = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t width_scale = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        // We are using shared memory to store weights wx, wy and a buffer of size wy unique per thread
        // Let's compute block_y size depending on given height_scale and width_scale
        // We have the following relationship:
        // shmem_size / sizeofdtype =
        //  interp_width * block_x +   <-- wx allocation
        //  interp_height * block_y * (block_x + 1)   <-- wy and buffer allocations

        auto interp_filter = InterpFilter();
        const int interp_height = 1 + 2 * (int)ceilf(
            (height_scale >= 1.0) ? interp_filter.size * 0.5 * height_scale : interp_filter.size * 0.5);
        const int interp_width = 1 + 2 * (int)ceilf(
            (width_scale >= 1.0) ? interp_filter.size * 0.5 * width_scale : interp_filter.size * 0.5);

        int numer = sharedMemPerBlock * 1.0 / sizeof(scalar_t) - interp_width * block_x;
        int denom = interp_height * (block_x + 1);
        int block_y = lastPow2((unsigned int) (numer / denom));
        block_y = std::min<int>(maxThreadsPerBlock / block_x, block_y);
        const dim3 block(block_x, block_y);

        int grid_y = std::min<int>(maxGridSize[1], ceil_div(output_height, block_y));
        int grid_z = std::min<int>(maxGridSize[2], input.size(0) * input.size(1));
        const dim3 grid(grid_x, grid_y, grid_z);

        // Compute actual size of required shared memory and verify if we can allocate it
        // - wx and wy size:
        size_t weights_per_block = interp_width * block_x + interp_height * block_y;
        // - buffer size:
        weights_per_block += interp_height * block_y * block_x;
        size_t shmem_size = weights_per_block * sizeof(scalar_t);
        TORCH_CHECK(
            shmem_size <= sharedMemPerBlock,
            "Provided interpolation parameters can not be handled with current algorithm implementation. ",
            "Please reduce the scale factor. Too much shared memory required: ",
            shmem_size, " vs ", sharedMemPerBlock);

        upsample_gen2d_aa_out_frame<scalar_t, accscalar_t>
            <<<grid,
               block,
               shmem_size,
               stream>>>(height_scale, width_scale, idata, odata, interp_filter);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  if (!output.is_contiguous()) {
      output.copy_(output_c);
  }
}

// In the code below interp_filter_t distinguishes between bilinear and bicubic interpolations
// InterpFilter as BilinearFilterFunctor <--> bilinear
// InterpFilter as BicubicFilterFunctor <--> bicubic
template<typename InterpFilter>
static void upsample_gen2d_aa_backward_out_cuda_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {

  // Inspired from UpSampleBicubic2d.cu::upsample_bicubic2d_backward_out_cuda_template
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_gen2d_backward_out_cuda", {grad_output_arg, grad_input_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input_size[2];
  int input_width = input_size[3];

  Tensor grad_output = grad_output_.contiguous();

  grad_input.zero_();

  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int* maxThreadsDim = at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
  int block_x = std::min<int>(maxThreadsDim[0], at::cuda::warp_size());
  int block_y = std::min<int>(maxThreadsDim[1], num_threads / block_x);
  const dim3 block(block_x, block_y);

  int* maxGridSize = at::cuda::getCurrentDeviceProperties()->maxGridSize;
  int grid_x = std::min<int>(maxGridSize[0], ceil_div(output_width, block_x));
  int grid_y = std::min<int>(maxGridSize[1], ceil_div(output_height, block_y));
  int grid_z = std::min<int>(maxGridSize[2], input_size[0] * input_size[1]);
  const dim3 grid(grid_x, grid_y, grid_z);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad_output.scalar_type(), "upsample_gen2d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor64<scalar_t, 4>();
        auto odata = grad_output.packed_accessor64<const scalar_t, 4>();

        const accscalar_t height_scale = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t width_scale = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        auto interp_filter = InterpFilter();
        const int interp_height = 1 + 2 * (int)ceilf(
            (height_scale >= 1.0) ? interp_filter.size * 0.5 * height_scale : interp_filter.size * 0.5);
        const int interp_width = 1 + 2 * (int)ceilf(
            (width_scale >= 1.0) ? interp_filter.size * 0.5 * width_scale : interp_filter.size * 0.5);

        size_t weights_per_block = interp_width * block_x + interp_height * block_y;
        size_t shmem_size = weights_per_block * sizeof(scalar_t);
        size_t sharedMemPerBlock = at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock;
        TORCH_CHECK(
            shmem_size <= sharedMemPerBlock,
            "Provided interpolation parameters can not be handled with current algorithm implementation. ",
            "Please reduce the scale factor. Too much shared memory required: ",
            shmem_size, " vs ", sharedMemPerBlock);

        upsample_gen2d_aa_backward_out_frame<scalar_t, accscalar_t>
            <<<grid,
               block,
               shmem_size,
               stream>>>(height_scale, width_scale, idata, odata, interp_filter);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace

TORCH_IMPL_FUNC(upsample_bilinear2d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output) {
  upsample_bilinear2d_out_cuda_template(output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_bilinear2d_backward_out_cuda");
  upsample_bilinear2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output) {

  upsample_gen2d_aa_out_cuda_template<upsample_antialias::BilinearFilterFunctor>(
      output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_bilinear2d_aa_backward_out_cuda");
  upsample_gen2d_aa_backward_out_cuda_template<upsample_antialias::BilinearFilterFunctor>(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

// We define bicubic anti-alias function implementations in this file instead of
// UpSampleBicubic2d.cu as we are using a single generic implementation
TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& output) {
  upsample_gen2d_aa_out_cuda_template<upsample_antialias::BicubicFilterFunctor>(
      output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    const Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_bicubic2d_aa_backward_out_cuda");
  upsample_gen2d_aa_backward_out_cuda_template<upsample_antialias::BicubicFilterFunctor>(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

} // namespace at::native
