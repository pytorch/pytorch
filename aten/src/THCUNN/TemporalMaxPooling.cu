#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"
#include "THCTensor.hpp"
#include "THCStorage.hpp"

#define TEMPORAL_MAX_POOLING_THREADS 1024

template <typename Dtype>
__global__ void cunn_TemporalMaxPooling_updateOutputKernel(Dtype *input, Dtype *output, THCIndex_t *indices, int64_t input_w, int64_t input_n, int64_t output_w, int64_t kW, int64_t dW) {
  // Block idx is the batch index, thread idx + block idx y * MAX_THREADS is the time index
  Dtype *input_data = input + blockIdx.x * input_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n * dW;
  Dtype *output_data = output + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;
  THCIndex_t *indices_data = indices + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;

  int64_t feat = 0;
  int64_t time = 0;
  int64_t max_time = input_n * kW;

  Dtype max_value;
  THCIndex_t max_index = 0;

  if (threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS < output_w) {
    // For all features
    for (feat = 0; feat < input_n; ++feat) {
      max_value = THCNumerics<Dtype>::min();
      // For all values in the kernel space
      for (time = 0; time < max_time; time += input_n) {
        if (max_value < input_data[time + feat]) {
          max_value = input_data[time + feat];
          max_index = time / input_n;
        }
      }
      output_data[feat] = max_value;
      indices_data[feat] = max_index;
    }
  }
}

template <typename Dtype>
__global__ void cunn_TemporalMaxPooling_updateGradInputKernel(Dtype *gradInput, Dtype *gradOutput, THCIndex_t *indices, int64_t input_w, int64_t input_n, int64_t output_w, int64_t kW, int64_t dW) {
  // Block idx is the batch index, thread idx + block idx y * MAX_THREADS is the time index
  Dtype *gradInput_data = gradInput + blockIdx.x * input_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n * dW;
  Dtype *gradOutput_data = gradOutput + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;
  THCIndex_t *indices_data = indices + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;

  int64_t feat = 0;

  if (threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS < output_w) {
    // For all features
    for (feat = 0; feat < input_n; ++feat) {
      gradInput_data[indices_data[feat] * input_n + feat] += gradOutput_data[feat];
    }
  }
}

template <typename Dtype>
__global__ void cunn_TemporalMaxPooling_updateGradInputKernelAtomic(Dtype *gradInput, Dtype *gradOutput, THCIndex_t *indices, int64_t input_w, int64_t input_n, int64_t output_w, int64_t kW, int64_t dW) {
  // Block idx is the batch index, thread idx + block idx y * MAX_THREADS is the time index
  Dtype *gradInput_data = gradInput + blockIdx.x * input_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n * dW;
  Dtype *gradOutput_data = gradOutput + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;
  THCIndex_t *indices_data = indices + blockIdx.x * output_w * input_n + (
      threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS) * input_n;

  int64_t feat = 0;

  if (threadIdx.x + blockIdx.y * TEMPORAL_MAX_POOLING_THREADS < output_w) {
    // For all features
    for (feat = 0; feat < input_n; ++feat) {
      atomicAdd(&gradInput_data[indices_data[feat] * input_n + feat], gradOutput_data[feat]);
    }
  }
}

#include "generic/TemporalMaxPooling.cu"
#include "THCGenerateFloatTypes.h"
