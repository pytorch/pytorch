#include "THCUNN.h"
#include "common.h"
#include "THCReduceApplyUtils.cuh"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include <thrust/functional.h>

#define MULTILABELMARGIN_THREADS 1024

template <typename Dtype, typename Acctype>
__global__ void cunn_MultiLabelMarginCriterion_updateOutput_kernel(Dtype *output,
                                                                   Dtype *input,
                                                                   THCIndex_t *target,
                                                                   Dtype *istarget,
                                                                   int nframe,
                                                                   int dim,
                                                                   int sizeaverage)
{
  // Temporary sums (for mapreduce)
  __shared__ Acctype sums[MULTILABELMARGIN_THREADS];

  // vectors:
  int k = blockIdx.x;
  Dtype *input_k = input + k*dim;
  THCIndex_t *target_k = target + k*dim;
  Dtype *output_k = output + k;
  Dtype *istarget_k = istarget + k*dim;

  // zero istarget
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    istarget_k[d] = ScalarConvert<int, Dtype>::to(0);
  }
  __syncthreads();

  // mark targets in istarget
  if (threadIdx.x == 0) {
    for (int dt = 0; dt < dim; dt++) {
      int target_idx = target_k[dt] - TH_INDEX_BASE;
      if (target_idx < 0) break;
      istarget_k[target_idx] = ScalarConvert<int, Dtype>::to(1);
    }
  }
  __syncthreads();

  // iterate over targets
  Acctype sum = 0;
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = target_k[dt] - TH_INDEX_BASE;
    if (target_idx < 0) break;

    // current value for target
    Dtype input_target_k = input_k[target_idx];

    // compare to all inputs (multithreaded):
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // contribute to loss only if not a target
      if (!ScalarConvert<Dtype, int>::to(istarget_k[d])) {
        Dtype z = 1 - input_target_k + input_k[d];
        if (z > 0)
          sum += z;
      }
    }
  }

  // reduce
  Acctype totalSum = reduceBlock(sums, blockDim.x, sum, thrust::plus<Acctype>(), (Acctype)0);
  if (threadIdx.x == 0) {
    if (sizeaverage) {
      *output_k = ScalarConvert<Acctype, Dtype>::to((totalSum / dim) / nframe);
    } else {
      *output_k = ScalarConvert<Acctype, Dtype>::to(totalSum / dim);
    }
  }
}

template <typename Dtype, typename Acctype>
__global__ void cunn_MultiLabelMarginCriterion_updateGradInput_kernel(Dtype *gradInput,
                                                                      Dtype *gradOutput,
                                                                      Dtype *input,
                                                                      THCIndex_t *target,
                                                                      Dtype *istarget,
                                                                      int nframe,
                                                                      int dim,
                                                                      int sizeaverage,
                                                                      int reduce)
{
  // Temporary sums (for mapreduce)
  __shared__ Acctype sums[MULTILABELMARGIN_THREADS];

  // vectors:
  int k = blockIdx.x;
  Dtype *input_k = input + k*dim;
  Dtype *gradInput_k = gradInput + k*dim;
  THCIndex_t *target_k = target + k*dim;
  Dtype *istarget_k = istarget + k*dim;
 
  Dtype *gradOutput_k = gradOutput;
  if (!reduce) {
    gradOutput_k += k;
  }

  // gain:
  Dtype g = ScalarConvert<Acctype, Dtype>::to( sizeaverage && reduce ? 1./((Acctype)(nframe*dim)) : 1./((Acctype)dim) );

  // zero gradients:
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    gradInput_k[d] = ScalarConvert<int, Dtype>::to(0);
  }
  __syncthreads();

  // iterate over targets
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = (int)target_k[dt] - TH_INDEX_BASE;
    if (target_idx < 0) break;

    // current value for target
    Dtype input_target_k = input_k[target_idx];

    // compare to all inputs (multithreaded):
    Acctype sum = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // contribute to loss only if not a target
      if (!ScalarConvert<Dtype, int>::to(istarget_k[d])) {
        Dtype z = 1 - input_target_k + input_k[d];
        if (z > 0) {
          sum -= g;
          gradInput_k[d] += g;
        }
      }
    }
    __syncthreads();

    // reduce sum
    Acctype totalSum = reduceBlock(sums, blockDim.x, sum, thrust::plus<Acctype>(), (Acctype)0);
    if (threadIdx.x == 0) {
      gradInput_k[target_idx] += ScalarConvert<Acctype, Dtype>::to(totalSum);
    }
  }

  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    gradInput_k[d] *= *gradOutput_k;
  }
}

#include "generic/MultiLabelMarginCriterion.cu"
#include "THCGenerateFloatTypes.h"

#undef MULTILABELMARGIN_THREADS
