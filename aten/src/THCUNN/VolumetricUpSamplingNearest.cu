#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THC/THCTensor.hpp>

#include <THCUNN/upsampling.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCAtomics.cuh>

template<typename Dtype, typename Acctype>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void nearest_neighbor_5d_kernel(
		const int n,
		const THCDeviceTensor<Dtype, 5> data1,
		THCDeviceTensor<Dtype, 5> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);
  const float depth_scale = (float) depth1 / (float) depth2;
  const float height_scale = (float) height1 / (float) height2;
  const float width_scale = (float) width1 / (float) width2;

  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int d2 = index / (height2*width2);            // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int d1 = d2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][d1][h1][w1];
          data2[n][c][d2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, width1);
    const int d1 = nearest_neighbor_compute_source_index(depth_scale, d2, depth1);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
	const Dtype val = data1[n][c][d1][h1][w1];
	data2[n][c][d2][h2][w2] = val;
      }
    }
  }
}

// Backward operation
template <typename Dtype, typename Acctype>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(1024)
#endif
__global__ void nearest_neighbor_5d_kernel_backward(
		const int n,
		THCDeviceTensor<Dtype, 5> data1,
		const THCDeviceTensor<Dtype, 5> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);
  const float depth_scale = (float) depth1 / (float) depth2;
  const float height_scale = (float) height1 / (float) height2;
  const float width_scale = (float) width1 / (float) width2;

  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int d2 = index / (height2*width2);            // 0:depth2-1

    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int d1 = d2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][d1][h1][w1];
          data1[n][c][d2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const int h1 = nearest_neighbor_compute_source_index(height_scale, h2, height1);
    const int w1 = nearest_neighbor_compute_source_index(width_scale, w2, width1);
    const int d1 = nearest_neighbor_compute_source_index(depth_scale, d2, depth1);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
	const Dtype val = data2[n][c][d2][h2][w2];
	atomicAdd(data1[n][c][d1][h1][w1].data(), val);
      }
    }
  }
}


#include <THCUNN/generic/VolumetricUpSamplingNearest.cu>
#include <THC/THCGenerateFloatTypes.h>
