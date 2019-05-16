#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/native/cuda/UpSample.cuh>

namespace at {
namespace native {
namespace {

template <typename scalar_t, typename accscalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void upsample_nearest2d_out_frame(
    const int n,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int height1 = idata.size(2);
  const int width1 = idata.size(3);
  const int height2 = odata.size(2);
  const int width2 = odata.size(3);

  const float height_scale = (float)height1 / (float)height2;
  const float width_scale = (float)width1 / (float)width2;

  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][h1][w1];
          odata[n][c][h2][w2] = val;
        }
      }
      return;
    }
    //
    const int h1 =
        nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int w1 =
        nearest_neighbor_compute_source_index(width_scale, w2, width1);

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t val = idata[n][c][h1][w1];
        odata[n][c][h2][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename scalar_t, typename accscalar_t>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void upsample_nearest2d_backward_out_frame(
    const int n,
    PackedTensorAccessor<scalar_t, 4> idata,
    const PackedTensorAccessor<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int height1 = idata.size(2);
  const int width1 = idata.size(3);
  const int height2 = odata.size(2);
  const int width2 = odata.size(3);

  const float height_scale = (float)height1 / (float)height2;
  const float width_scale = (float)width1 / (float)width2;

  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = odata[n][c][h2][w2];
          idata[n][c][h1][w1] = val;
        }
      }
      return;
    }
    //
    const int h1 =
        nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int w1 =
        nearest_neighbor_compute_source_index(width_scale, w2, width1);

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[n][c][h2][w2];
        atomicAdd(&idata[n][c][h1][w1], d2val);
      }
    }
  }
}

static void upsample_nearest2d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(
      "upsample_nearest2d_out_cuda_template", {input_arg, output_arg});

  AT_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  upsample_2d_shape_check(
      input,
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

  output.resize_({input.size(0), input.size(1), output_height, output_width});
  output.zero_();

  const int num_kernels = output_height * output_width;
  const int num_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_nearest2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 4>();
        auto odata = output.packed_accessor<scalar_t, 4>();

        upsample_nearest2d_out_frame<scalar_t, accscalar_t>
            <<<cuda::ATenCeilDiv(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(num_kernels, idata, odata);
      });

      AT_CUDA_CHECK(cudaGetLastError());
}

static void upsample_nearest2d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_nearest2d_backward_out_cuda",
      {grad_output_arg, grad_input_arg});

  AT_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  AT_CHECK(
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

  grad_input.zero_();

  const int num_kernels = output_height * output_width;
  const int num_threads =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_nearest2d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor<scalar_t, 4>();
        auto odata = grad_output.packed_accessor<scalar_t, 4>();

        upsample_nearest2d_backward_out_frame<scalar_t, accscalar_t>
            <<<cuda::ATenCeilDiv(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(num_kernels, idata, odata);
      });

      AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

Tensor& upsample_nearest2d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  upsample_nearest2d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor upsample_nearest2d_cuda(const Tensor& input, IntArrayRef output_size) {
  Tensor output = at::empty_like(input);
  upsample_nearest2d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor& upsample_nearest2d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  upsample_nearest2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

Tensor upsample_nearest2d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  Tensor grad_input = at::empty_like(grad_output);
  upsample_nearest2d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

} // namespace native
} // namespace at
