#include "THCUNN.h"
#include "common.h"
#include "THCTensor.hpp"

#include "linear_upsampling.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

template<typename Dtype, typename Acctype>
__global__ void nearest_neighbor_interp2_kernel(
		const int n,
		const bool align_corners,
		const THCDeviceTensor<Dtype, 3> data1,
		THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  const Acctype scale = width1 / width2;

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
	for (int c = 0; c < channels; ++c) {
	  const Dtype val = data1[n][c][w1];
	  data2[n][c][w2] = val;
	}
      }
      return;
    }
    //
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1, align_corners);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
	const Dtype val = data1[n][c][w1];
	data2[n][c][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename Dtype, typename Acctype>
__global__ void nearest_neighbor_interp2_kernel_backward(
		const int n, 
		const bool align_corners,
		THCDeviceTensor<Dtype, 3> data1,
		const THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  const Acctype scale = width1 / width2;

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
	for (int c = 0; c < channels; ++c) {
	  const Dtype val = data2[n][c][w1];
	  data1[n][c][w2] = val;
	}
      }
      return;
    }
    //
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1, align_corners);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
	      const Dtype d2val = data2[n][c][w2];
	      atomicAdd(data1[n][c][w1].data(), d2val);
      }
    }
  }
}

/*
 * Description:
 */

/*
__device__ int translate_idx(int ii, int d1, int d2, int scale_factor)
{
  int x, y, z;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  z = z/scale_factor;
  d2 /= scale_factor;
  return ((x*d1+y)*d2)+z;

}
__device__ int translate_idx_inv(int ii, int d1, int d2, int scale_factor, int off_x)
{
  int x, y, z;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  z = z*scale_factor+off_x;
  d2 *= scale_factor;
  return ((x*d1+y)*d2)+z;

}

template <typename Dtype>
__global__ void upscale(Dtype *input, Dtype *output, int64_t no_elements,
                        int scale_factor, int d1, int d2)
{
  // output offset:
  int64_t ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, scale_factor);
  output[ii]=input[ipidx];
}
*/

/*
 * Description:
 */
/*
template <typename Dtype, typename Acctype>
__global__ void downscale(Dtype *gradInput_data, Dtype *gradOutput_data, int64_t no_elements,
                              int scale_factor, int d1, int d2)
{
  // output offset:
  int64_t ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  Acctype sum = Acctype(0);
  for (int i=0; i < scale_factor; i++){
    int ipidx = translate_idx_inv(ii, d1, d2, scale_factor, i);
    sum += gradOutput_data[ipidx];
  }
  gradInput_data[ii] += ScalarConvert<Acctype, Dtype>::to(sum);
}
*/

#include "generic/TemporalUpSamplingNearest.cu"
#include "THCGenerateFloatTypes.h"
