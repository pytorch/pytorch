#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THC/THCTensor.hpp>

#include <THCUNN/linear_upsampling.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCAtomics.cuh>

template<typename Dtype, typename Acctype>
__global__ void nearest_neighbor_3d_kernel(
		const int n,
		const THCDeviceTensor<Dtype, 3> data1,
		THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  const float scale = (float) width1 / (float) width2;

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
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1);
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
__global__ void nearest_neighbor_3d_kernel_backward(
		const int n,
		THCDeviceTensor<Dtype, 3> data1,
		const THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  const float scale = (float) width1 / (float) width2;

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
    const int w1 = nearest_neighbor_compute_source_index(scale, w2, width1);
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][w2];
        atomicAdd(data1[n][c][w1].data(), d2val);
      }
    }
  }
}


#include <THCUNN/generic/TemporalUpSamplingNearest.cu>
#include <THC/THCGenerateFloatTypes.h>
