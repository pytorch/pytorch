// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#include "THCUNN.h"
#include "common.h"
#include "linear_upsampling.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

template<typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel(const int n,
    const Acctype rwidth, const bool align_corners,
    const THCDeviceTensor<Dtype, 3> data1, THCDeviceTensor<Dtype, 3> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);

  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][w1];
          data2[n][c][w2] = val;
        }
      }
      return;
    }
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
        const Acctype val = w0lambda * data1[n][c][w1]
                            + w1lambda * data1[n][c][w1+w1p];
        data2[n][c][w2] = ScalarConvert<Acctype, Dtype>::to(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, typename Acctype>
__global__ void caffe_gpu_interp2_kernel_backward(const int n,
    const Acctype rwidth, const bool align_corners,
    THCDeviceTensor<Dtype, 3> data1, const THCDeviceTensor<Dtype, 3> data2){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int width1 = data1.getSize(2);
  const int width2 = data2.getSize(2);
  if (index < n) {
    const int w2 = index % width2;
    // special case: just copy
    if (width1 == width2) {
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][w1];
          data1[n][c][w2] += val;
        }
      }
      return;
    }
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
      for (int c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][w2];
        atomicAdd(data1[n][c][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(w0lambda * d2val));
        atomicAdd(data1[n][c][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(w1lambda * d2val));
      }
    }
  }
}


#include "generic/TemporalUpSamplingLinear.cu"
#include "THCGenerateFloatTypes.h"
