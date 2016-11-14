#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#define SOFTMAX_THREADS 128

template <typename T, typename AccumT>
__global__ void cunn_SoftMax_updateOutput_kernel(
  T *output, T *input, int nframe, int dim, int stride0, int stride1)
{
  __shared__ AccumT buffer[SOFTMAX_THREADS+1];
  T *input_k  = input  + blockIdx.x*dim*stride0 + blockIdx.y*stride1 + blockIdx.z;
  T *output_k = output + blockIdx.x*dim*stride0 + blockIdx.y*stride1 + blockIdx.z;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -THCNumerics<AccumT>::max();
  for (int i=i_start; i<i_end; i+=i_step)
  {
    T z = input_k[i*stride0];
    AccumT zAcc = ScalarConvert<T, AccumT>::to(z);
    if (buffer[threadIdx.x] < zAcc)
      buffer[threadIdx.x] = zAcc;
  }


  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    AccumT max_k = -THCNumerics<AccumT>::max();
    for (int i=0; i<blockDim.x; i++)
    {
      if (max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[SOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // sum?
  T max_k = ScalarConvert<AccumT, T>::to(buffer[SOFTMAX_THREADS]);
  buffer[threadIdx.x] = ScalarConvert<int, AccumT>::to(0);
  for (int i=i_start; i<i_end; i+=i_step) {
    T z = THCNumerics<T>::exp(input_k[i*stride0]-max_k);
    buffer[threadIdx.x] += ScalarConvert<T, AccumT>::to(z);
    output_k[i*stride0] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    AccumT sum_k = ScalarConvert<int, AccumT>::to(0);
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[SOFTMAX_THREADS] = sum_k;
  }

  __syncthreads();

  // softmax
  T sum_k = ScalarConvert<AccumT, T>::to(buffer[SOFTMAX_THREADS]);
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i*stride0] = output_k[i*stride0] / sum_k;
}

template <typename T, typename AccumT>
__global__ void cunn_SoftMax_updateGradInput_kernel(
  T *gradInput, T *output, T *gradOutput, int nframe, int dim, int stride0, int stride1)
{
  __shared__ AccumT buffer[SOFTMAX_THREADS];
  T *gradInput_k  = gradInput  + blockIdx.x*dim*stride0 + blockIdx.y * stride1 + blockIdx.z;
  T *output_k     = output     + blockIdx.x*dim*stride0 + blockIdx.y * stride1 + blockIdx.z;
  T *gradOutput_k = gradOutput + blockIdx.x*dim*stride0 + blockIdx.y * stride1 + blockIdx.z;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // sum?
  buffer[threadIdx.x] = ScalarConvert<int, AccumT>::to(0);
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += ScalarConvert<T, AccumT>::to(gradOutput_k[i*stride0] * output_k[i*stride0]);

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    AccumT sum_k = ScalarConvert<int, AccumT>::to(0);
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[0] = sum_k;
  }

  __syncthreads();

  T sum_k = ScalarConvert<AccumT, T>::to(buffer[0]);
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i*stride0] = output_k[i*stride0] * (gradOutput_k[i*stride0] - sum_k);
}

#include "generic/SoftMax.cu"
#include "THCGenerateFloatTypes.h"

#undef SOFTMAX_THREADS
