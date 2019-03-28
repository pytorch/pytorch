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
