#include "THCUNN.h"
#include "common.h"
#include "THCTensor.hpp"

#include "linear_upsampling.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

template<typename Dtype, typename Acctype>
__global__ void nearest_neighbor_3d_kernel(
		const int64_t n,
		const THCDeviceTensor<Dtype, 3> data1,
		THCDeviceTensor<Dtype, 3> data2) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t batchsize = data1.getSize(0);
  const int64_t channels = data1.getSize(1);
  const int64_t width1 = data1.getSize(2);
  const int64_t width2 = data2.getSize(2);
  const float scale = (float) width1 / (float) width2;

  if (index < n) {
    const int64_t w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int64_t w1 = w2;
      for (int64_t n = 0; n < batchsize; n++) {
	for (int64_t c = 0; c < channels; ++c) {
	  const Dtype val = data1[n][c][w1];
	  data2[n][c][w2] = val;
	}
      }
      return;
    }
    //
    const int64_t w1 = nearest_neighbor_compute_source_index(scale, w2, width1);
    for (int64_t n = 0; n < batchsize; n++) {
      for (int64_t c = 0; c < channels; ++c) {
	const Dtype val = data1[n][c][w1];
	data2[n][c][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename Dtype, typename Acctype>
__global__ void nearest_neighbor_3d_kernel_backward(
		const int64_t n,
		THCDeviceTensor<Dtype, 3> data1,
		const THCDeviceTensor<Dtype, 3> data2) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t batchsize = data1.getSize(0);
  const int64_t channels = data1.getSize(1);
  const int64_t width1 = data1.getSize(2);
  const int64_t width2 = data2.getSize(2);
  const float scale = (float) width1 / (float) width2;

  if (index < n) {
    const int64_t w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int64_t w1 = w2;
      for (int64_t n = 0; n < batchsize; n++) {
	for (int64_t c = 0; c < channels; ++c) {
	  const Dtype val = data2[n][c][w1];
	  data1[n][c][w2] = val;
	}
      }
      return;
    }
    //
    const int64_t w1 = nearest_neighbor_compute_source_index(scale, w2, width1);
    for (int64_t n = 0; n < batchsize; n++) {
      for (int64_t c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][w2];
        atomicAdd(data1[n][c][w1].data(), d2val);
      }
    }
  }
}


#include "generic/TemporalUpSamplingNearest.cu"
#include "THCGenerateFloatTypes.h"
