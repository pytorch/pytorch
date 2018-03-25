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
__launch_bounds__(1024)
__global__ void caffe_gpu_interp2_kernel(const int n,
    const Acctype rdepth, const Acctype rheight, const Acctype rwidth, const bool align_corners,
    const THCDeviceTensor<Dtype, 5> data1, THCDeviceTensor<Dtype, 5> data2) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);

  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int t2 = index / (height2*width2);            // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data1[n][c][t1][h1][w1];
          data2[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }
    //
    const Acctype t1r = linear_upsampling_compute_source_index<Acctype>(rdepth, t2, align_corners);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const Acctype t1lambda = t1r - t1;
    const Acctype t0lambda = Acctype(1) - t1lambda;
    //
    const Acctype h1r = linear_upsampling_compute_source_index<Acctype>(rheight, h2, align_corners);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
        const Acctype val = t0lambda * (h0lambda * (w0lambda * data1[n][c][t1][h1][w1]
                                                  + w1lambda * data1[n][c][t1][h1][w1+w1p])
                                      + h1lambda * (w0lambda * data1[n][c][t1][h1+h1p][w1]
                                                  + w1lambda * data1[n][c][t1][h1+h1p][w1+w1p]))
                          + t1lambda * (h0lambda * (w0lambda * data1[n][c][t1+t1p][h1][w1]
                                                  + w1lambda * data1[n][c][t1+t1p][h1][w1+w1p])
                                      + h1lambda * (w0lambda * data1[n][c][t1+t1p][h1+h1p][w1]
                                                  + w1lambda * data1[n][c][t1+t1p][h1+h1p][w1+w1p]));
        data2[n][c][t2][h2][w2] = ScalarConvert<Acctype, Dtype>::to(val);
      }
    }
  }
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, typename Acctype>
__launch_bounds__(1024)
__global__ void caffe_gpu_interp2_kernel_backward(const int n,
    const Acctype rdepth, const Acctype rheight, const Acctype rwidth, const bool align_corners,
    THCDeviceTensor<Dtype, 5> data1, const THCDeviceTensor<Dtype, 5> data2){
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  const int batchsize = data1.getSize(0);
  const int channels = data1.getSize(1);
  const int depth1 = data1.getSize(2);
  const int height1 = data1.getSize(3);
  const int width1 = data1.getSize(4);
  const int depth2 = data2.getSize(2);
  const int height2 = data2.getSize(3);
  const int width2 = data2.getSize(4);
  if (index < n) {
    const int w2 = (index % (height2*width2)) % width2; // 0:width2-1
    const int h2 = (index % (height2*width2)) / width2; // 0:height2-1
    const int t2 = index / (height2*width2);            // 0:depth2-1
    // special case: just copy
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize ; n++){
        for (int c = 0; c < channels; ++c) {
          const Dtype val = data2[n][c][t1][h1][w1];
          data1[n][c][t2][h2][w2] += val;
        }
      }
      return;
    }
    //
    const Acctype t1r = linear_upsampling_compute_source_index<Acctype>(rdepth, t2, align_corners);
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const Acctype t1lambda = t1r - t1;
    const Acctype t0lambda = Acctype(1) - t1lambda;
    //
    const Acctype h1r = linear_upsampling_compute_source_index<Acctype>(rheight, h2, align_corners);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Acctype h1lambda = h1r - h1;
    const Acctype h0lambda = Acctype(1) - h1lambda;
    //
    const Acctype w1r = linear_upsampling_compute_source_index<Acctype>(rwidth, w2, align_corners);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const Acctype w1lambda = w1r - w1;
    const Acctype w0lambda = Acctype(1) - w1lambda;
    //
    for (int n = 0; n < batchsize ; n++){
      for (int c = 0; c < channels; ++c) {
        const Dtype d2val = data2[n][c][t2][h2][w2];
        atomicAdd(data1[n][c][t1][h1][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h0lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1][h1][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h0lambda * w1lambda * d2val));
        atomicAdd(data1[n][c][t1][h1+h1p][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h1lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1][h1+h1p][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t0lambda * h1lambda * w1lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h0lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h0lambda * w1lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1+h1p][w1].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h1lambda * w0lambda * d2val));
        atomicAdd(data1[n][c][t1+t1p][h1+h1p][w1+w1p].data(),
                  ScalarConvert<Acctype, Dtype>::to(t1lambda * h1lambda * w1lambda * d2val));
      }
    }
  }
  /////////////////////////////////////////////////////////
}


#include "generic/VolumetricUpSamplingTrilinear.cu"
#include "THCGenerateFloatTypes.h"
