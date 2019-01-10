#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"

#include <stdio.h>
#include <assert.h>

static const int NTHREADS = 32;

template <typename Dtype>
__global__ void cunn_ClassNLLCriterion_updateOutput_kernel1(Dtype *output,
                                                           Dtype *total_weight,
                                                           Dtype *input,
                                                           THCIndex_t  *target,
                                                           Dtype *weights,
                                                           int size_average,
                                                           int n_classes,
                                                           int64_t ignore_index) {
  assert(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel.

  int t = (int) *target - TH_INDEX_BASE;
  if (t != (int) ignore_index) {
    assert(t >= 0 && t < n_classes);
    Dtype cur_weight = weights ? weights[t] : ScalarConvert<int, Dtype>::to(1);
    *output = -cur_weight * input[t];
    *total_weight = cur_weight;
    if (size_average && *total_weight > 0) {
      *output /= *total_weight;
    }
  }
}

template <typename Dtype>
__global__ void ClassNLLCriterion_updateOutput_no_reduce_kernel(
    int batch_size,
    THCDeviceTensor<Dtype, 2> input,
    THCDeviceTensor<THCIndex_t, 1> target,
    THCDeviceTensor<Dtype, 1> output,
    Dtype *weights,
    int n_classes,
    int ignore_index) {

  CUDA_KERNEL_LOOP(index, batch_size) {
    int cur_target = target[index] - TH_INDEX_BASE;
    if (cur_target == ignore_index) {
      output[index] = ScalarConvert<int, Dtype>::to(0);
      continue;
    }
    assert(cur_target  >= 0 && cur_target  < n_classes);
    Dtype weight =
       weights ? weights[cur_target] : ScalarConvert<int, Dtype>::to(1);
    output[index] = -weight * input[index][cur_target];
  }
}

template <typename Dtype>
__global__ void ClassNLLCriterion_updateGradInput_no_reduce_kernel(
    int batch_size,
    THCDeviceTensor<THCIndex_t, 1> target,
    THCDeviceTensor<Dtype, 1> gradOutput,
    THCDeviceTensor<Dtype, 2> gradInput,
    Dtype *weights,
    int n_classes,
    int ignore_index) {

  CUDA_KERNEL_LOOP(index, batch_size) {
    int cur_target = target[index] - TH_INDEX_BASE;
    if (cur_target == ignore_index) {
      continue;
    }
    assert(cur_target  >= 0 && cur_target  < n_classes);
    Dtype weight =
       weights ? weights[cur_target] : ScalarConvert<int, Dtype>::to(1);
    gradInput[index][cur_target] = -weight * gradOutput[index];
  }
}

template <typename Dtype, typename Acctype>
__global__ void cunn_ClassNLLCriterion_updateOutput_kernel(Dtype *output,
                                                           Dtype *total_weight,
                                                           Dtype *input,
                                                           THCIndex_t *target,
                                                           Dtype *weights,
                                                           int size_average,
                                                           int nframe,
                                                           int ndim,
                                                           int n_classes,
                                                           int64_t ignore_index) {
  __shared__ Acctype shInputs[NTHREADS], acc_weight[NTHREADS];
  int i, t;
  Dtype cur_weight;

  shInputs[threadIdx.x] = ScalarConvert<int, Acctype>::to(0);
  acc_weight[threadIdx.x] = ScalarConvert<int, Acctype>::to(0);
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
      t = target[i] - TH_INDEX_BASE;
      if (t != (int) ignore_index) {
        assert(t >= 0 && t < n_classes);
        cur_weight = weights ? weights[t] : ScalarConvert<int, Dtype>::to(1);
        shInputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
        acc_weight[threadIdx.x] += cur_weight;
      }
  }
  __syncthreads();

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel

  if (threadIdx.x == 0) {
    *output = *total_weight = ScalarConvert<int, Dtype>::to(0);
    Acctype outputAcc = 0;
    Acctype total_weightAcc = 0;
    for (i = 0; i < NTHREADS; ++i){
      // FIXME should we do somethigng here
      outputAcc += shInputs[i];
      total_weightAcc += acc_weight[i];
    }
    *total_weight = ScalarConvert<Acctype, Dtype>::to(total_weightAcc);
    *output = ScalarConvert<Acctype, Dtype>::to(outputAcc);
    if (size_average && *total_weight > 0) {
      *output = ScalarConvert<Acctype, Dtype>::to(outputAcc / total_weightAcc);
    }

  }
}

template <typename Dtype>
__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel1(
  Dtype* gradInput,
  Dtype* gradOutput,
  Dtype* weights,
  THCIndex_t* target,
  Dtype* total_weight,
  int size_average,
  int n_classes,
  int64_t ignore_index)
{
  if (*total_weight <= 0) {
    return;
  }
  Dtype norm = size_average ? (ScalarConvert<int, Dtype>::to(1) / *total_weight) : ScalarConvert<int, Dtype>::to(1);
  int t = (int)*target - TH_INDEX_BASE;
  if (t != (int) ignore_index) {
    assert(t >= 0 && t < n_classes);
    gradInput[t] = -(weights ? weights[t] : ScalarConvert<int, Dtype>::to(1)) * norm * gradOutput[0];
  }
}

template <typename Dtype>
__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel(
  Dtype *gradInput,
  Dtype *gradOutput,
  THCIndex_t *target,
  Dtype *weights,
  Dtype *total_weight,
  int size_average,
  int nframe,
  int ndim,
  int n_classes,
  int64_t ignore_index)
{
  if (*total_weight <= 0) {
    return;
  }
  int i, t;
  Dtype norm = size_average ? (ScalarConvert<int, Dtype>::to(1) / *total_weight) : ScalarConvert<int, Dtype>::to(1);

  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    t = (int)target[i] - TH_INDEX_BASE;
    if (t != (int) ignore_index) {
      assert(t >= 0 && t < n_classes);
      gradInput[i * ndim + t] = -(weights ? weights[t] : ScalarConvert<int, Dtype>::to(1)) * norm * gradOutput[0];
    }
  }
}

#include "generic/ClassNLLCriterion.cu"
#include "THCGenerateFloatTypes.h"
