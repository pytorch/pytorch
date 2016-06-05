#include "THCUNN.h"
#include "common.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <thrust/unique.h>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

const int WARP_SIZE = 32;

__device__ __forceinline__ bool warpHasCollision(int val)
{
  // Compare our value to the values stored in the next 16 lanes,
  // wrapping around at 32. If any pair of values is the same than
  // there is a collision in the warp.
  bool dup = 0;
  const int laneId = threadIdx.x % 32;

#if __CUDA_ARCH__ >= 300

  #pragma unroll
  for (int i = 1; i <= 16; i++)
  {
    dup |= (__shfl(val, (laneId + i) % 32) == val);
  }

#else

  volatile __shared__ int values[128];
  values[threadIdx.x] = val;
  const int offset = threadIdx.x - laneId;

  #pragma unroll
  for (int i = 1; i <= 16; i++)
  {
    dup |= (values[offset + ((laneId + i) % 32)] == val);
  }

#endif

  return __any(dup) != 0;
}

__global__ void cunn_LookupTable_accGradParametersKernelByFeature(
  float *input, float *gradOutput, float *gradWeight, float scale, long numel,
  long stride, int paddingValue) {

  const int featureDim = blockIdx.x * 4 + threadIdx.x / 32;
  if (featureDim >= stride) {
    return;
  }

  // The strategy here is that each warp handles a single feature
  // dimension.
  // Within that feature dimension, points in the [batch][element]
  // dimension can overlap, and we need to determine if threads want
  // to add to the gradient in a colliding manner.
  // Typically one would use floating-point atomicAdd() to resolve
  // these collisions, but that is non-deterministic if there are
  // collisions. Non-determinism for this code is really bad,
  // especially in RNNs, and is prone to snowballing error.
  // In order to get a deterministic order of execution, we handle
  // non-colliding updates separately from colliding ones. Colliding
  // updates are serialized in their order of execution by using the
  // warp-wide collision detector `warpHasCollision`.
  const int laneId = threadIdx.x % 32;
  for (int i = laneId; i < numel; i += WARP_SIZE) {
    const int weightIndex = (int) (input[i] - 1);
    if (weightIndex == paddingValue - 1) {
      continue;
    }

    float update = gradOutput[i*stride + featureDim] * scale;

    // Check for collision
    if (warpHasCollision(weightIndex)) {
      // Run all lanes sequentially; warp divergence
      for (int i = 0; i < WARP_SIZE; ++i) {
        if (laneId == i) {
          gradWeight[weightIndex*stride + featureDim] += update;
        }
      }
    } else {
      // No collision; warp coherence
      gradWeight[weightIndex*stride + featureDim] += update;
    }
  }
}

__global__ void cunn_LookupTable_accGradParametersKernel(
  float *input, float *indices, float *gradOutput, float *gradWeight,
  float *count, float defaultScale, long numel, long stride, int paddingValue) {

  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceeding input has the same as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values proceessed by each thread (grain size)
  const int SZ = 4;

  if (idx < numel
      && (idx == 0 || input[idx] != input[idx - 1])
      && input[idx] != paddingValue) {
    do {
      const int startFeature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weightRow = ((int) input[idx] - 1) * stride;
      const int gradOutputRow = ((int) indices[idx] - 1) * stride;
      const float scale = count ? defaultScale / count[idx] : defaultScale;

      float gradient[SZ];
      float weight[SZ];

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          gradient[ii] = gradOutput[gradOutputRow + featureDim];
          weight[ii] = gradWeight[weightRow + featureDim];
        }
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        weight[ii] += gradient[ii] * scale;
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          gradWeight[weightRow + featureDim] = weight[ii];
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

void THNN_CudaLookupTable_accGradParameters(
  THCState *state,
  THIndexTensor *input,
  THCudaTensor *gradOutput,
  THCudaTensor *gradWeight,
  THIntegerTensor *count,
  THCudaTensor *sorted,
  THCudaTensor *indices,
  bool scaleGradByFreq,
  int paddingValue,
  float scale)
{
  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, sorted, indices);
  if (!(THCudaTensor_isContiguous(state, input) &&
        THCudaTensor_isContiguous(state, gradOutput) &&
        THCudaTensor_isContiguous(state, gradWeight)))
  {
    THError("Tensors must be contiguous");
  }

  int nDim = THCudaTensor_nDimension(state, input);
  if (nDim != 1 && nDim != 2)
    THError("input must be a vector or matrix");

  long numel = THCudaTensor_nElement(state, input);
  long stride = gradWeight->stride[0];

  cudaStream_t stream = THCState_getCurrentStream(state);

  if (numel <= 768 && !scaleGradByFreq) {
    cunn_LookupTable_accGradParametersKernelByFeature<<<DIVUP(stride,4), 128, 0, stream>>>(
      THCudaTensor_data(state, input),
      THCudaTensor_data(state, gradOutput),
      THCudaTensor_data(state, gradWeight),
      scale,
      numel,
      stride,
      paddingValue);
    THCudaCheck(cudaGetLastError());
    return;
  }

  THCudaTensor_resizeAs(state, sorted, input);
  THCudaTensor_resizeAs(state, indices, input);

  // Sort the inputs into sorted with the corresponding indices
  THCudaTensor_sort(state, sorted, indices, input, 0, 0);

  float *sorted_data = THCudaTensor_data(state, sorted);
  float *indices_data = THCudaTensor_data(state, indices);
  float *count_data = NULL;

  if (scaleGradByFreq)
  {
    THIntegerTensor_(resizeAs)(state, count, input);
    count_data = THIntegerTensor_(data)(state, count);

    thrust::device_ptr<float> sorted_ptr(sorted_data);
    thrust::device_ptr<float> count_ptr(count_data);

    // Compute an increasing sequence per unique item in sorted:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 1 2 3 1 2 1 1 2
    thrust::inclusive_scan_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      sorted_ptr,
      sorted_ptr + numel,
      thrust::make_constant_iterator(1),
      count_ptr
    );

    // Take the maximum of each count per unique key in reverse:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    thrust::inclusive_scan_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      thrust::make_reverse_iterator(sorted_ptr + numel),
      thrust::make_reverse_iterator(sorted_ptr),
      thrust::make_reverse_iterator(count_ptr + numel),
      thrust::make_reverse_iterator(count_ptr + numel),
      thrust::equal_to<float>(),
      thrust::maximum<float>()
    );
  }

  dim3 grid(DIVUP(numel,4), DIVUP(stride,128));
  dim3 block(32, 4);
  cunn_LookupTable_accGradParametersKernel<<<grid, block, 0, stream>>>(
    sorted_data,
    indices_data,
    THCudaTensor_data(state, gradOutput),
    THCudaTensor_data(state, gradWeight),
    count_data,
    scale,
    numel,
    stride,
    paddingValue
  );
  THCudaCheck(cudaGetLastError());
}

/*
 * Keep the norm of weight smaller than maxNorm
 */
template <typename T>
struct pow_v
{
  T normType;
  pow_v(T v) : normType(v) {}
  __host__ __device__
  T operator()(const T& x) const {
    if (normType == 1)
      return std::abs(x);
    else if (normType == 2)
      return x * x;
    else
      return std::pow(std::abs(x), normType);
  }
};

template <typename T>
struct multiply_s
{
  T scale;
  multiply_s(T s) : scale(s) {}
  __host__ __device__
  T operator()(const T& x) const {
    return x * scale;
  }
};

void THNN_CudaLookupTable_renorm(
  THCState *state,
  THIndexTensor *idx,
  THCudaTensor *weight,
  float maxNorm,
  float normType)
{
  THCUNN_assertSameGPU(state, 2, idx, weight);
  if (!(THCudaTensor_isContiguous(state, idx) &&
        THCudaTensor_isContiguous(state, weight)))
  {
    THError("Tensors must be contiguous");
  }
  if (THCudaTensor_nDimension(state, idx) != 1)
    THError("idx must be a vector");
  if (normType <= 0)
    THError("non-positive-norm not supported");

  long numel = THCudaTensor_nElement(state, idx);
  long stride = weight->stride[0];

  // get the unique indices
  thrust::device_ptr<float> weight_ptr(THCudaTensor_data(state, weight));
  thrust::device_ptr<float> idx_ptr(THCudaTensor_data(state, idx));
  thrust::device_ptr<float> end_ptr = thrust::unique(idx_ptr, idx_ptr+numel);
  numel = end_ptr - idx_ptr;

  pow_v<float> unary_pow(normType);
  thrust::plus<float> binary_plus;
  // numel << stride, since idx usually contains sparse row indices
  for (long i = 0; i < numel; i++)
  {
    long k = idx_ptr[i] - 1;
    thrust::device_ptr<float> row_ptr = weight_ptr + k * stride;
    float norm = thrust::transform_reduce(row_ptr, row_ptr + stride,
      unary_pow, 0, binary_plus);
    norm = std::pow(norm, 1.0 / normType);
    if (norm > maxNorm)
    {
      multiply_s<float> unary_mul(maxNorm / (norm + 1e-7));
      thrust::transform(row_ptr, row_ptr + stride, row_ptr, unary_mul);
    }
  }
}
