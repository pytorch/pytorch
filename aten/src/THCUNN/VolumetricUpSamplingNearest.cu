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
__global__ void nearest_neighbor_5d_kernel(
		const int64_t n,
		const THCDeviceTensor<Dtype, 5> data1,
		THCDeviceTensor<Dtype, 5> data2) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t batchsize = data1.getSize(0);
  const int64_t channels = data1.getSize(1);
  const int64_t depth1 = data1.getSize(2);
  const int64_t height1 = data1.getSize(3);
  const int64_t width1 = data1.getSize(4);
  const int64_t depth2 = data2.getSize(2);
  const int64_t height2 = data2.getSize(3);
  const int64_t width2 = data2.getSize(4);
  const float depth_scale = (float) depth1 / (float) depth2;
  const float height_scale = (float) height1 / (float) height2;
  const float width_scale = (float) width1 / (float) width2;

  if (index < n) {
    const int64_t w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int64_t h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int64_t d2 = index / (height2*width2);            // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int64_t d1 = d2;
      const int64_t h1 = h2;
      const int64_t w1 = w2;
      for (int64_t n = 0; n < batchsize ; n++){
        for (int64_t c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][d1][h1][w1];
          data2[n][c][d2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const int64_t h1 = nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int64_t w1 = nearest_neighbor_compute_source_index(width_scale, w2, width1);
    const int64_t d1 = nearest_neighbor_compute_source_index(depth_scale, d2, depth1);
    for (int64_t n = 0; n < batchsize; n++) {
      for (int64_t c = 0; c < channels; ++c) {
	const Dtype val = data1[n][c][d1][h1][w1];
	data2[n][c][d2][h2][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename Dtype, typename Acctype>
__global__ void nearest_neighbor_5d_kernel_backward(
		const int64_t n,
		THCDeviceTensor<Dtype, 5> data1,
		const THCDeviceTensor<Dtype, 5> data2) {
  int64_t index = threadIdx.x + blockIdx.x * blockDim.x;
  const int64_t batchsize = data1.getSize(0);
  const int64_t channels = data1.getSize(1);
  const int64_t depth1 = data1.getSize(2);
  const int64_t height1 = data1.getSize(3);
  const int64_t width1 = data1.getSize(4);
  const int64_t depth2 = data2.getSize(2);
  const int64_t height2 = data2.getSize(3);
  const int64_t width2 = data2.getSize(4);
  const float depth_scale = (float) depth1 / (float) depth2;
  const float height_scale = (float) height1 / (float) height2;
  const float width_scale = (float) width1 / (float) width2;

  if (index < n) {
    const int64_t w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int64_t h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int64_t d2 = index / (height2*width2);            // 0:depth2-1

    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int64_t d1 = d2;
      const int64_t h1 = h2;
      const int64_t w1 = w2;
      for (int64_t n = 0; n < batchsize ; n++){
        for (int64_t c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][d1][h1][w1];
          data1[n][c][d2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const int64_t h1 = nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int64_t w1 = nearest_neighbor_compute_source_index(width_scale, w2, width1);
    const int64_t d1 = nearest_neighbor_compute_source_index(depth_scale, d2, depth1);
    for (int64_t n = 0; n < batchsize; n++) {
      for (int64_t c = 0; c < channels; ++c) {
	const Dtype val = data2[n][c][d2][h2][w2];
	atomicAdd(data1[n][c][d1][h1][w1].data(), val);
      }
    }
  }
}


#include "generic/VolumetricUpSamplingNearest.cu"
#include "THCGenerateFloatTypes.h"
