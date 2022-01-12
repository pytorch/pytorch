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

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/upsample_bilinear2d_native.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at {
namespace native {
namespace {

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_out_frame(
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int height1 = idata.size(2);
  const int width1 = idata.size(3);
  const int height2 = odata.size(2);
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
    const int batchsize,
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
    const size_t nc,
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
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  const auto memory_format = input.suggest_memory_format();

  if (input.sizes() == output.sizes()) {
    output.copy_(input);
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
    // heuristic: only use channels_last path when it's faster than the contiguous path
    if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 16 && \
          output.is_contiguous(memory_format)) {
      using accscalar_t = at::acc_type<scalar_t, true>;

      TORCH_CHECK(input.numel() < std::numeric_limits<int>::max(),
        "upsample_bilinear2d_nhwc only supports input tensors with less than INT_MAX elements");
      TORCH_CHECK(output.numel() < std::numeric_limits<int>::max(),
        "upsample_bilinear2d_nhwc only supports output tensors with less than INT_MAX elements");

      const int batchsize = input.size(0);
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

      const scalar_t* idata = input_cl.data_ptr<scalar_t>();
      scalar_t* odata = output.data_ptr<scalar_t>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_height, output_height, align_corners, scales_h);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_width, output_width, align_corners, scales_w);

      upsample_bilinear2d_nhwc_out_frame<scalar_t, accscalar_t>
        <<<ceil_div(num_kernels, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          rheight, rwidth, align_corners,
          batchsize,
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

      auto idata = input.packed_accessor64<scalar_t, 4>();
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
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output_.scalar_type(), "upsample_bilinear2d_backward_out_frame", [&] {
    if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 && \
          grad_input.is_contiguous(memory_format)) {
      using accscalar_t = at::acc_type<scalar_t, true>;

      Tensor grad_output = grad_output_.contiguous(at::MemoryFormat::ChannelsLast);

      auto idata = grad_input.data_ptr<scalar_t>();
      auto odata = grad_output.data_ptr<scalar_t>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_height, output_height, align_corners, scales_h);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_width, output_width, align_corners, scales_w);

      upsample_bilinear2d_backward_nhwc_out_frame<scalar_t, accscalar_t>
          <<<ceil_div(num_kernels, static_cast<size_t>(num_threads)), num_threads, 0, stream>>>(
              nbatch * channels,
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

      auto idata = grad_input_c.data_ptr<scalar_t>();
      auto odata = grad_output.data_ptr<scalar_t>();

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

} // namespace

TORCH_IMPL_FUNC(upsample_bilinear2d_out_cuda) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output) {
  upsample_bilinear2d_out_cuda_template(output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_cuda) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_bilinear2d_backward_out_cuda");
  upsample_bilinear2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

} // namespace native
} // namespace at
