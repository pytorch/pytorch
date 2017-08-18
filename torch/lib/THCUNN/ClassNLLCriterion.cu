#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

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
                                                           long ignore_index) {
  assert(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel.

  int t = (int)*target - TH_INDEX_BASE;
  if (t != ignore_index) {
    assert(t >= 0 && t < n_classes);
    Dtype cur_weight = weights ? weights[t] : ScalarConvert<int, Dtype>::to(1);
    *output = -cur_weight * input[t];
    *total_weight = cur_weight;
    if (size_average && *total_weight > 0) {
      *output /= *total_weight;
    }
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
                                                           long ignore_index) {
  __shared__ Acctype shInputs[NTHREADS], acc_weight[NTHREADS];
  int i, t;
  Dtype cur_weight;

  shInputs[threadIdx.x] = ScalarConvert<int, Acctype>::to(0);
  acc_weight[threadIdx.x] = ScalarConvert<int, Acctype>::to(0);
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
      t = target[i] - TH_INDEX_BASE;
      if (t != ignore_index) {
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
  Dtype* weights,
  THCIndex_t* target,
  Dtype* total_weight,
  int size_average,
  int n_classes,
  long ignore_index)
{
  if (*total_weight <= 0) {
    return;
  }
  Dtype norm = size_average ? (ScalarConvert<int, Dtype>::to(1) / *total_weight) : ScalarConvert<int, Dtype>::to(1);
  int t = (int)*target - TH_INDEX_BASE;
  if (t != ignore_index) {
    assert(t >= 0 && t < n_classes);
    gradInput[t] = -(weights ? weights[t] : ScalarConvert<int, Dtype>::to(1)) * norm;
  }
}

template <typename Dtype>
__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel(
  Dtype *gradInput,
  THCIndex_t *target,
  Dtype *weights,
  Dtype *total_weight,
  int size_average,
  int nframe,
  int ndim,
  int n_classes,
  long ignore_index)
{
  if (*total_weight <= 0) {
    return;
  }
  int i, t;
  Dtype norm = size_average ? (ScalarConvert<int, Dtype>::to(1) / *total_weight) : ScalarConvert<int, Dtype>::to(1);

  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    t = (int)target[i] - TH_INDEX_BASE;
    if (t != ignore_index) {
      assert(t >= 0 && t < n_classes);
      gradInput[i * ndim + t] = -(weights ? weights[t] : ScalarConvert<int, Dtype>::to(1)) * norm;
    }
  }
}

#include "generic/ClassNLLCriterion.cu"
#include "THCGenerateFloatTypes.h"
