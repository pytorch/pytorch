// Copyright 2004-present Facebook. All Rights Reserved.
#ifndef CAFFE2_UTILS_MIXED_UTILS_H
#define CAFFE2_UTILS_MIXED_UTILS_H

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

// define functions to allow add/mult/store operaions for input/output with
// mixed precisions.
namespace caffe2 {

// functions that will only be triggered when there is no spcialized version
// supported
template <typename T, typename T2>
inline __device__ T mixed_mult(T data1, T2 data2) {
  return data1 * data2;
};

template <typename T, typename T2>
inline __device__ T mixed_add(T data1, T2 data2) {
  return data1 + data2;
};

template <typename TIN, typename TOUT>
inline __device__ void mixed_store(TIN* data_in, TOUT* data_out) {
  *data_out = *data_in;
  return;
};

template <typename T>
inline __device__ void mixed_store(T* data_in, T* data_out) {
  *data_out = *data_in;
  return;
};

#ifdef CAFFE_HAS_CUDA_FP16
// define templated functions to support mixed precision computation
template <>
inline __device__ float mixed_mult(float data1, const float data2) {
  return data1 * data2;
}

template <>
inline __device__ float mixed_mult(float data1, const half data2) {
  return data1 * __half2float(data2);
}

template <>
inline __device__ float mixed_mult(float data1, float16 data2) {
  half* data2_half = reinterpret_cast<half*>(&data2);
  return data1 * __half2float(*data2_half);
}
template <>
inline __device__ float mixed_add(float data1, const float data2) {
  return data1 + data2;
}

template <>
inline __device__ float mixed_add(float data1, const half data2) {
  return data1 + __half2float(data2);
}

template <>
inline __device__ float mixed_add(float data1, float16 data2) {
  half* data2_half = reinterpret_cast<half*>(&data2);
  return data1 + __half2float(*data2_half);
}

template <>
inline __device__ void mixed_store(float* data_in, float* data_out) {
  *data_out = *data_in;
  return;
}

template <>
inline __device__ void mixed_store(half* data_in, float* data_out) {
  *data_out = __half2float(*data_in);
  return;
}

template <>
inline __device__ void mixed_store(float16* data_in, float* data_out) {
  half* data_in_half = reinterpret_cast<half*>(data_in);
  *data_out = __half2float(*data_in_half);
  return;
}

template <>
inline __device__ void mixed_store(float* data_in, float16* data_out) {
  half data_in_half = __float2half(*data_in);
  float16* data_in_float16 = reinterpret_cast<float16*>(&data_in_half);
  *data_out = *data_in_float16;
  return;
}

template <>
inline __device__ void mixed_store(float* data_in, half* data_out) {
  half data_in_half = __float2half(*data_in);
  *data_out = data_in_half;
  return;
}
#endif // for CAFFE_HAS_CUDA_FP16
} // namespace caffe2
#endif // for CAFFE2_UTILS_MIXED_UTILS_H
