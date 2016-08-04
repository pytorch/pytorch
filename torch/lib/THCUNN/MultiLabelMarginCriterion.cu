#include "THCUNN.h"
#include "common.h"
#include "THCReduceApplyUtils.cuh"

#include <thrust/functional.h>

#define MULTILABELMARGIN_THREADS 1024

__global__ void cunn_MultiLabelMarginCriterion_updateOutput_kernel(float *output,
                                                                   float *input,
                                                                   float *target,
                                                                   float *istarget,
                                                                   int nframe,
                                                                   int dim,
                                                                   int sizeaverage)
{
  // Temporary sums (for mapreduce)
  __shared__ float sums[MULTILABELMARGIN_THREADS];

  // vectors:
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *target_k = target + k*dim;
  float *output_k = output + k;
  float *istarget_k = istarget + k*dim;

  // zero istarget
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    istarget_k[d] = 0;
  }
  __syncthreads();

  // mark targets in istarget
  if (threadIdx.x == 0) {
    for (int dt = 0; dt < dim; dt++) {
      int target_idx = (int)target_k[dt];
      if (target_idx == 0) break;
      istarget_k[target_idx - 1] = 1;
    }
  }
  __syncthreads();

  // iterate over targets
  float sum = 0;
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = (int)target_k[dt];
    if (target_idx == 0) break;

    // current value for target
    float input_target_k = input_k[target_idx-1];

    // compare to all inputs (multithreaded):
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // contribute to loss only if not a target
      if (!istarget_k[d]) {
        float z = 1 - input_target_k + input_k[d];
        if (z > 0)
          sum += z;
      }
    }
  }

  // reduce
  float totalSum = reduceBlock(sums, blockDim.x, sum, thrust::plus<float>(), 0.0f);
  if (threadIdx.x == 0) {
    if (sizeaverage) {
      *output_k = (totalSum / dim) / nframe;
    } else {
      *output_k = totalSum / dim;
    }
  }
}

__global__ void cunn_MultiLabelMarginCriterion_updateGradInput_kernel(float *gradInput,
                                                                      float *input,
                                                                      float *target,
                                                                      float *istarget,
                                                                      int nframe,
                                                                      int dim,
                                                                      int sizeaverage)
{
  // Temporary sums (for mapreduce)
  __shared__ float sums[MULTILABELMARGIN_THREADS];

  // vectors:
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *gradInput_k = gradInput + k*dim;
  float *target_k = target + k*dim;
  float *istarget_k = istarget + k*dim;

  // gain:
  float g = ( sizeaverage ? 1./((float)(nframe*dim)) : 1./((float)dim) );

  // zero gradients:
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    gradInput_k[d] = 0;
  }
  __syncthreads();

  // iterate over targets
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = (int)target_k[dt];
    if (target_idx == 0) break;

    // current value for target
    float input_target_k = input_k[target_idx-1];

    // compare to all inputs (multithreaded):
    float sum = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // contribute to loss only if not a target
      if (!istarget_k[d]) {
        float z = 1 - input_target_k + input_k[d];
        if (z > 0) {
          sum -= g;
          gradInput_k[d] += g;
        }
      }
    }
    __syncthreads();

    // reduce sum
    float totalSum = reduceBlock(sums, blockDim.x, sum, thrust::plus<float>(), 0.0f);
    if (threadIdx.x == 0) {
      gradInput_k[target_idx-1] += totalSum;
    }
    __syncthreads();
  }
}

void THNN_CudaMultiLabelMarginCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          THCudaTensor *istarget,
          bool sizeaverage)
{
  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);
  istarget = THCudaTensor_newContiguous(state, istarget);
  THCudaTensor_resizeAs(state, istarget, input);

  if(input->nDimension == 1)
  {
    THCudaTensor_resize1d(state, output, 1);

    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<<<blocks,threads>>>(
        THCudaTensor_data(state, output),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        THCudaTensor_data(state, istarget),
        1, input->size[0],
        sizeaverage
        );
    THCudaCheck(cudaGetLastError());
  }
  else if(input->nDimension == 2)
  {
    THCudaTensor *output_tmp = THCudaTensor_newWithSize1d(state, input->size[0]);

    dim3 blocks(input->size[0]);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<<<blocks,threads>>>(
        THCudaTensor_data(state, output_tmp),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        THCudaTensor_data(state, istarget),
        input->size[0], input->size[1],
        sizeaverage
        );
    THCudaCheck(cudaGetLastError());
    THCudaTensor_resize1d(state, output, 1);
    THCudaTensor_set1d(state, output, 0, THCudaTensor_sumall(state, output_tmp));
    THCudaTensor_free(state, output_tmp);
  }
  else
    THError("vector or matrix expected");

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, istarget);
}

void THNN_CudaMultiLabelMarginCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          THCudaTensor *istarget,
          bool sizeaverage)
{
  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);
  istarget = THCudaTensor_newContiguous(state, istarget);
  THCudaTensor_resizeAs(state, gradInput, input);

  if(gradInput->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        THCudaTensor_data(state, istarget),
        1, gradInput->size[0],
        sizeaverage);

  }
  else if(gradInput->nDimension == 2)
  {
    dim3 blocks(gradInput->size[0]);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        THCudaTensor_data(state, istarget),
        gradInput->size[0], gradInput->size[1],
        sizeaverage);
  }
  else
    THError("vector or matrix expected");

  THCudaCheck(cudaGetLastError());

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, istarget);
}

#undef MULTILABELMARGIN_THREADS