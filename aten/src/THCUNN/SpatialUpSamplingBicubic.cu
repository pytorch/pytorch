#include "THCUNN.h"
#include "THCTensor.hpp"
#include "common.h"
#include "linear_upsampling.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

__constant__ int size = (1 << 10);

template<typename Dtype>
__device__ Dtype access_tensor(
  const THCDeviceTensor<Dtype, 4> data,
  int channel,
  int batch,
  int width,
  int height,
  int x,
  int y
) {
  int access_x = max(min(x, width - 1), 0);
  int access_y = max(min(y, height - 1), 0);
  return data[batch][channel][access_y][access_x];
}

template<typename Dtype, typename Acctype>
__device__ void inc_tensor(
  const THCDeviceTensor<Dtype, 4> data,
  int channel,
  int batch,
  int width,
  int height,
  int x,
  int y,
  Acctype value
) {
  int access_x = max(min(x, width - 1), 0);
  int access_y = max(min(y, height - 1), 0);
  atomicAdd(
    data[batch][channel][access_y][access_x].data(),
    ScalarConvert<Acctype, Dtype>::to(value)
  );
}

template<typename Acctype>
__device__ Acctype bicubic_convolution1(Acctype x, Acctype A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template<typename Acctype>
__device__ Acctype bicubic_convolution2(Acctype x, Acctype A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template<typename Acctype>
__device__ void THNN_(get_coefficients)(
  Acctype coeffs[4],
  Acctype t,
  int size
) {
  int offset = t * size;
  Acctype A = -0.75;

  Acctype x1 = offset * 1.0 / size;
  coeffs[0] = bicubic_convolution2<Acctype>(x1 + 1.0, A);
  coeffs[1] = bicubic_convolution1<Acctype>(x1, A);

  // opposite coefficients
  Acctype x2 = (size - offset) * 1.0 / size;
  coeffs[2] = bicubic_convolution1<Acctype>(x2, A);
  coeffs[3] = bicubic_convolution2<Acctype>(x2 + 1.0, A);
}

template<typename Dtype, typename Acctype>
__device__ static Acctype cubic_interp1d(
  Dtype x0,
  Dtype x1,
  Dtype x2,
  Dtype x3,
  Acctype t,
  int size
) {
  Acctype coeffs[4];
  THNN_(get_coefficients)<Acctype>(coeffs, t, size);

  return x0 * coeffs[0]
    + x1 * coeffs[1]
    + x2 * coeffs[2]
    + x3 * coeffs[3];
}


template<typename Dtype, typename Acctype>
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
          access_tensor<Dtype>(
            in_data, c, n, input_width, input_height, in_x - 1, in_y - 1 + k),
          access_tensor<Dtype>(
            in_data, c, n, input_width, input_height, in_x + 0, in_y - 1 + k),
          access_tensor<Dtype>(
            in_data, c, n, input_width, input_height, in_x + 1, in_y - 1 + k),
          access_tensor<Dtype>(
            in_data, c, n, input_width, input_height, in_x + 2, in_y - 1 + k),
          t_x,
          size
        );
      }

      out_data[n][c][output_y][output_x] = ScalarConvert<Acctype, Dtype>::to(cubic_interp1d(
        coefficients[0],
        coefficients[1],
        coefficients[2],
        coefficients[3],
        t_y,
        size
      ));
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, typename Acctype>
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

  THNN_(get_coefficients)(x_coeffs, t_x, size);
  THNN_(get_coefficients)(y_coeffs, t_y, size);

  for (int n = 0; n < batchsize ; n++){
    for (int c = 0; c < channels; ++c) {
      Dtype out_value = out_data[n][c][output_y][output_x];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          inc_tensor<Dtype, Acctype>(
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


#include "generic/SpatialUpSamplingBicubic.cu"
#include "THCGenerateFloatTypes.h"
