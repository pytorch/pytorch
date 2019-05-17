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
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest1d_out_frame(
    const int n,
    const PackedTensorAccessor<scalar_t, 3> idata,
    PackedTensorAccessor<scalar_t, 3> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int width1 = idata.size(2);
  const int width2 = odata.size(2);

  const float scale = (float)width1 / (float)width2;

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
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1);

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t val = idata[n][c][w1];
        odata[n][c][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_nearest1d_backward_out_frame(
    const int n,
    PackedTensorAccessor<scalar_t, 3> idata,
    const PackedTensorAccessor<scalar_t, 3> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int width1 = idata.size(2);
  const int width2 = odata.size(2);

  const float scale = (float)width1 / (float)width2;

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
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1);

    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const scalar_t d2val = odata[n][c][w2];
        atomicAdd(&idata[n][c][w1], d2val);
      }
    }
  }
}

static void upsample_nearest1d_out_cuda_template(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_nearest1d_out_cuda", {input_arg, output_arg});

  AT_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  int output_width = output_size[0];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_width = input.size(2);

  upsample_1d_shape_check(
      input, Tensor(), nbatch, channels, input_width, output_width);

  AT_ASSERT(input_width > 0 && output_width > 0);

  output.resize_({input.size(0), input.size(1), output_width});
  output.zero_();

  const int num_kernels = output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_nearest1d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 3>();
        auto odata = output.packed_accessor<scalar_t, 3>();

        upsample_nearest1d_out_frame<scalar_t, accscalar_t>
            <<<cuda::ATenCeilDiv(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(num_kernels, idata, odata);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

static void upsample_nearest1d_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_nearest1d_backward_out_cuda_template",
      {grad_output_arg, grad_input_arg});

  AT_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  AT_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int output_width = output_size[0];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_width = input_size[2];

  upsample_1d_shape_check(
      Tensor(), grad_output_, nbatch, channels, input_width, output_width);

  Tensor grad_output = grad_output_.contiguous();
  grad_input.resize_({nbatch, channels, input_width});
  grad_input.zero_();

  const int num_kernels = output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "upsample_nearest1d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor<scalar_t, 3>();
        auto odata = grad_output.packed_accessor<scalar_t, 3>();

        upsample_nearest1d_backward_out_frame<scalar_t, accscalar_t>
            <<<cuda::ATenCeilDiv(num_kernels, num_threads),
               num_threads,
               0,
               stream>>>(num_kernels, idata, odata);
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

} // namespace

Tensor& upsample_nearest1d_out_cuda(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  upsample_nearest1d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor upsample_nearest1d_cuda(const Tensor& input, IntArrayRef output_size) {
  Tensor output = at::empty_like(input);
  upsample_nearest1d_out_cuda_template(output, input, output_size);
  return output;
}

Tensor& upsample_nearest1d_backward_out_cuda(
    Tensor& grad_input,
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  upsample_nearest1d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

Tensor upsample_nearest1d_backward_cuda(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size) {
  Tensor grad_input = at::empty_like(grad_output);
  upsample_nearest1d_backward_out_cuda_template(
      grad_input, grad_output, output_size, input_size);
  return grad_input;
}

} // namespace native
} // namespace at
