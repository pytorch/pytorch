#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"
#include "common.h"
#include <THC/THCApply.cuh>

#include <thrust/functional.h>

template <typename T, typename AccumT>
__global__ void cunn_SpatialClassNLLCriterion_updateOutput_kernel(
          T *output,
          T *total_weight,
          T *input,
          THCIndex_t *target,
          T *weights,
          int size_average,
          int batch_size,
          int n_classes,
          int map_nelem,
          int blocks_per_sample,
          long ignore_index)
{
  __shared__ AccumT partial_sums[CUDA_NUM_THREADS];

  int i, t;
  T cur_weight;
  AccumT input_sum = 0;
  AccumT acc_weight = 0;

  int sample = blockIdx.x / blocks_per_sample;
  int toffset = sample * map_nelem;
  int ioffset = sample * map_nelem * n_classes;
  int step = blockDim.x * blocks_per_sample;
  for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x;
       i < map_nelem;
       i += step) {
    t = target[toffset + i] - TH_INDEX_BASE;
    if (t != ignore_index) {
      assert(t >= 0 && t < n_classes);
      cur_weight = weights ? weights[t] : ScalarConvert<int, T>::to(1);
      input_sum -= input[ioffset + i + map_nelem * t] * cur_weight;
      acc_weight += cur_weight;
    }
  }

  __syncthreads();

  input_sum = reduceBlock(partial_sums, blockDim.x, input_sum, thrust::plus<AccumT>(), AccumT(0));
  acc_weight = reduceBlock(partial_sums, blockDim.x, acc_weight, thrust::plus<AccumT>(), AccumT(0));

  if (threadIdx.x == 0) {
    atomicAdd(total_weight, ScalarConvert<AccumT, T>::to(acc_weight));
    atomicAdd(output, ScalarConvert<AccumT, T>::to(input_sum));
  }
}

template<typename T>
__global__ void cunn_SpatialClassNLLCriterion_sizeAverage_kernel(
          T *output,
          T *total_weight)
{
  if (*total_weight > 0)
    *output = THCNumerics<T>::div(*output, *total_weight);
}

template<typename T>
__global__ void cunn_SpatialClassNLLCriterion_updateGradInput_kernel(
          T *gradInput,
          THCIndex_t *target,
          T *weights,
          T *total_weight,
          int size_average,
          int batch_size,
          int n_classes,
          int map_nelem,
          int blocks_per_sample,
          long ignore_index)
{
  if (*total_weight <= 0)
    return;

  int i, t;
  T norm = size_average ? (ScalarConvert<int, T>::to(1) / *total_weight) : ScalarConvert<int, T>::to(1);

  int sample = blockIdx.x / blocks_per_sample;
  int step = blockDim.x * blocks_per_sample;
  int toffset = sample * map_nelem;
  int ioffset = sample * map_nelem * n_classes;
  for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x;
       i < map_nelem;
       i += step) {
    t = (int)target[toffset + i] - TH_INDEX_BASE;
    if (t != ignore_index) {
      assert(t >= 0 && t < n_classes);
      gradInput[ioffset + i + map_nelem * t] = -(weights ? weights[t] : ScalarConvert<int, T>::to(1)) * norm;
    }
  }
}

#include "generic/SpatialClassNLLCriterion.cu"
#include "THCGenerateFloatTypes.h"
