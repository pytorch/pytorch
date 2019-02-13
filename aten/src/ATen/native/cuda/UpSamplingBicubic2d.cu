#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>
#include <THCUNN/upsampling.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCAtomics.cuh>

template<typename Dtype, typename Acctype>
#if defined(__HIP_PLATFORM_HCC__)
__launch_bounds__(1024)
#endif
__global__ void bicubic_interp2d_kernel(
  const int num_elements,
  const Acctype height_scale,
  const Acctype width_scale,
  const THCDeviceTensor<Dtype, 4> in_data,
  THCDeviceTensor<Dtype, 4> out_data
) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = in_data.getSize(0);
  const int channels = in_data.getSize(1);
  const int input_height = in_data.getSize(2);
  const int input_width = in_data.getSize(3);
  const int output_height = out_data.getSize(2);
  const int output_width = out_data.getSize(3);

  if (index >= num_elements) {
    return;
  }

  // Special case: input and output are the same size, just copy
  const int output_x = index % output_width;
  const int output_y = index / output_width;
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++){
      for (int c = 0; c < channels; c++) {
        const Dtype val = in_data[n][c][output_y][output_x];
        out_data[n][c][output_x][output_y] = val;
      }
    }
    return;
  }

  // Interpolation kernel
  Acctype real_x = width_scale * output_x;
  int in_x = real_x;
  Acctype t_x = real_x - in_x;

  Acctype real_y = height_scale * output_y;
  int in_y = real_y;
  Acctype t_y = real_y - in_y;

  for (int n = 0; n < batchsize ; n++) {
    for (int c = 0; c < channels; c++) {
      Acctype coefficients[4];

      for (int k = 0; k < 4; k++) {
        coefficients[k] = cubic_interp1d(
          upsampling_get_value_bounded<Dtype>(
            in_data, c, n, input_width, input_height, in_x - 1, in_y - 1 + k),
          upsampling_get_value_bounded<Dtype>(
            in_data, c, n, input_width, input_height, in_x + 0, in_y - 1 + k),
          upsampling_get_value_bounded<Dtype>(
            in_data, c, n, input_width, input_height, in_x + 1, in_y - 1 + k),
          upsampling_get_value_bounded<Dtype>(
            in_data, c, n, input_width, input_height, in_x + 2, in_y - 1 + k),
          t_x
        );
      }

      out_data[n][c][output_y][output_x] = ScalarConvert<Acctype, Dtype>::to(cubic_interp1d(
        coefficients[0],
        coefficients[1],
        coefficients[2],
        coefficients[3],
        t_y
      ));
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, typename Acctype>
#if defined(__HIP_PLATFORM_HCC__)
__launch_bounds__(1024)
#endif
__global__ void bicubic_interp2d_backward_kernel(
  const int num_elements,
  const Acctype height_scale,
  const Acctype width_scale,
  const bool align_corners,
  THCDeviceTensor<Dtype, 4> in_data,
  const THCDeviceTensor<Dtype, 4> out_data
){

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = in_data.getSize(0);
  const int channels = in_data.getSize(1);
  const int input_height = in_data.getSize(2);
  const int input_width = in_data.getSize(3);
  const int output_height = out_data.getSize(2);
  const int output_width = out_data.getSize(3);

  if (index >= num_elements) {
    return;
  }

  const int output_x = index % output_width;
  const int output_y = index / output_width;
  // special case: output_xust copy
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize ; n++){
      for (int c = 0; c < channels; ++c) {
        const Dtype val = out_data[n][c][output_y][output_x];
        in_data[n][c][output_y][output_x] += val;
      }
    }
    return;
  }

  Acctype real_x = width_scale * output_x;
  int input_x = real_x;
  Acctype t_x = real_x - input_x;

  Acctype real_y = height_scale * output_y;
  int input_y = real_y;
  Acctype t_y = real_y - input_y;

  Acctype x_coeffs[4];
  Acctype y_coeffs[4];

  get_cubic_upsampling_coefficients(x_coeffs, t_x);
  get_cubic_upsampling_coefficients(y_coeffs, t_y);

  for (int n = 0; n < batchsize ; n++){
    for (int c = 0; c < channels; ++c) {
      Dtype out_value = out_data[n][c][output_y][output_x];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          upsampling_increment_value_bounded<Dtype, Acctype>(
            in_data,
            c,
            n,
            input_width,
            input_height,
            input_x - 1 + j,
            input_y - 1 + i,
            out_value * y_coeffs[i] * x_coeffs[j]
          );
        }
      }
    }
  }
}


#include <THCUNN/generic/SpatialUpSamplingBicubic.cu>
#include <THC/THCGenerateFloatTypes.h>

// -----------

#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THCUNN/generic/SpatialUpSamplingBicubic.cu"
#else

#include <THCUNN/upsampling.h>
#include <ATen/cuda/CUDAContext.h>

static inline void THNN_(SpatialUpSamplingBicubic_shapeCheck)
                        (THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         int nBatch, int nChannels,
                         int inputHeight, int inputWidth,
                         int outputHeight, int outputWidth) {
  THArgCheck(inputHeight > 0 && inputWidth > 0
             && outputHeight > 0 && outputWidth > 0, 2,
             "input and output sizes should be greater than 0,"
             " but got input (H: %d, W: %d) output (H: %d, W: %d)",
             inputHeight, inputWidth, outputHeight, outputWidth);
  if (input != NULL) {
     THCUNN_argCheck(state, !input->is_empty() && input->dim() == 4, 2, input,
                     "non-empty 4D input tensor expected but got: %s");
  }

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, 4, 0, nBatch);
    THCUNN_check_dim_size(state, gradOutput, 4, 1, nChannels);
    THCUNN_check_dim_size(state, gradOutput, 4, 2, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, 4, 3, outputWidth);
  }
}

void THNN_(SpatialUpSamplingBicubic_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int outputHeight,
           int outputWidth,
           bool align_corners)
{
  int nbatch = THCTensor_(size)(state, input, 0);
  int channels = THCTensor_(size)(state, input, 1);
  int inputHeight = THCTensor_(size)(state, input, 2);
  int inputWidth = THCTensor_(size)(state, input, 3);
  THNN_(SpatialUpSamplingBicubic_shapeCheck)
       (state, input, NULL,
        nbatch, channels,
        inputHeight, inputWidth,
        outputHeight, outputWidth);

  THCUNN_assertSameGPU(state, 2, input, output);
  THCTensor_(resize4d)(state, output,
                       THCTensor_(size)(state, input, 0),
                       THCTensor_(size)(state, input, 1),
                       outputHeight, outputWidth);
  THCTensor_(zero)(state, output);
  THCDeviceTensor<scalar_t, 4> idata = toDeviceTensor<scalar_t, 4>(state, input);
  THCDeviceTensor<scalar_t, 4> odata = toDeviceTensor<scalar_t, 4>(state, output);
  THAssert(inputHeight > 0 && inputWidth > 0 && outputHeight > 0 && outputWidth > 0);

  // Get scaling factors
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);

  const int num_output_elements = outputHeight * outputWidth;
  const int max_threads =
    at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  // Launch kernel
  cudaStream_t stream = THCState_getCurrentStream(state);
  bicubic_interp2d_kernel<scalar_t, accreal> <<<
    THCCeilDiv(num_output_elements, max_threads),
    max_threads,
    0,
    stream
  >>>(num_output_elements, rheight, rwidth, idata, odata);
  THCudaCheck(cudaGetLastError());
}


void THNN_(SpatialUpSamplingBicubic_updateGradInput)(
           THCState *state,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           int nbatch,
           int nchannels,
           int inputHeight,
           int inputWidth,
           int outputHeight,
           int outputWidth,
           bool align_corners)
{
  THNN_(SpatialUpSamplingBicubic_shapeCheck)
       (state, NULL, gradOutput,
        nbatch, nchannels,
        inputHeight, inputWidth,
        outputHeight, outputWidth);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  THCUNN_assertSameGPU(state, 2, gradOutput, gradInput);
  THCTensor_(resize4d)(state, gradInput, nbatch, nchannels, inputHeight, inputWidth);
  THCTensor_(zero)(state, gradInput);
  THCDeviceTensor<scalar_t, 4> in_data = toDeviceTensor<scalar_t, 4>(state, gradInput);
  THCDeviceTensor<scalar_t, 4> out_data = toDeviceTensor<scalar_t, 4>(state, gradOutput);
  const accreal rheight = linear_upsampling_compute_scale<accreal>(inputHeight, outputHeight, align_corners);
  const accreal rwidth = linear_upsampling_compute_scale<accreal>(inputWidth, outputWidth, align_corners);
  const int num_kernels = outputHeight * outputWidth;
  const int num_threads =
    at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  cudaStream_t stream = THCState_getCurrentStream(state);
  bicubic_interp2d_backward_kernel<scalar_t ,accreal> <<<THCCeilDiv(num_kernels, num_threads),
  num_threads, 0, stream>>>(num_kernels, rheight, rwidth, align_corners, in_data, out_data);
  THCudaCheck(cudaGetLastError());
  THCTensor_(free)(state, gradOutput);
}

#endif
