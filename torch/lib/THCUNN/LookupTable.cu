#include "THCUNN.h"
#include "common.h"
#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <thrust/unique.h>
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCTensorSort.cuh"

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

template <typename Dtype>
__global__ void cunn_LookupTable_accGradParametersKernelByFeature(
  int64_t *input, Dtype *gradOutput, Dtype *gradWeight, Dtype scale, ptrdiff_t numel,
  int64_t stride, int paddingValue) {

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
  for (ptrdiff_t i = laneId; i < numel; i += WARP_SIZE) {
    const int weightIndex = (int) (input[i] - TH_INDEX_BASE);
    if (weightIndex == paddingValue - TH_INDEX_BASE) {
      continue;
    }

    Dtype update = gradOutput[i*stride + featureDim] * scale;

    // FIXME: should we accumulate as accreal?
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

template <typename Dtype, typename Acctype>
__global__ void cunn_LookupTable_accGradParametersKernel(
  int64_t *input, int64_t *indices, Dtype *gradOutput, Dtype *gradWeight,
  int64_t *count, Dtype defaultScale, ptrdiff_t numel, int64_t stride, int paddingValue) {

  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceding input has the same as this input, then the warp
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
      const int weightRow = ((int) input[idx] - TH_INDEX_BASE) * stride;
      const int gradOutputRow = ((int) indices[idx] - TH_INDEX_BASE) * stride;
      const Acctype scale = count ? ScalarConvert<Dtype, Acctype>::to(defaultScale) / count[idx] : ScalarConvert<Dtype, Acctype>::to(defaultScale);

      Acctype gradient[SZ];
      Acctype weight[SZ];

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          gradient[ii] = ScalarConvert<Dtype, Acctype>::to(gradOutput[gradOutputRow + featureDim]);
          weight[ii] = ScalarConvert<Dtype, Acctype>::to(gradWeight[weightRow + featureDim]);
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
          gradWeight[weightRow + featureDim] = ScalarConvert<Acctype, Dtype>::to(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

template <typename Dtype, typename Acctype>
__device__
Acctype fast_pow(Dtype x, Acctype normType) {
    Acctype xA = ScalarConvert<Dtype, Acctype>::to(x);
    if (normType == 1)
      return std::abs(xA);
    else if (normType == 2)
      return xA * xA;
    else
      return std::pow(std::abs(xA), normType);
}

template <typename Dtype>
__device__
Dtype get_padded_value(Dtype *weight_ptr, THCIndex_t idx, int64_t dim) 
{
  if (idx < dim)
    return weight_ptr[idx];
  else
    return 0;
}

/* Calculate norms of the rows of weight_ptr given by idx_ptr and capture them in norms */
template <typename Dtype, typename Acctype>
__global__
void calculate_norms_and_renorm(Dtype *weight_ptr, THCIndex_t *idx_ptr, Dtype normType, 
    Dtype maxNorm, Acctype normType_, int64_t dim)
{
  // Some casting hacks since dynamic shared memory and templates don't work together:
  extern __shared__ __align__(sizeof(Acctype)) unsigned char smem[];
  Acctype *sdata = reinterpret_cast<Acctype *>(smem);

  int64_t tid = threadIdx.x;
  int64_t idx = (idx_ptr[blockIdx.x] - TH_INDEX_BASE) * dim + tid;

  // Do the initial exponents in the load
  // Shenanigans here since we need to work with a power-of-2 block size
  // effectively pad the matrix with zeroes if needed
  Acctype second;
  if (tid + blockDim.x > dim)
    second = (Acctype) 0;
  else 
    second = fast_pow(weight_ptr[idx + blockDim.x], normType_);

  sdata[tid] = fast_pow(weight_ptr[idx], normType_) + second;

  __syncthreads();

  // iterative merging algorithm. Possible optimization could involve 
  for(int64_t i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i) {
      sdata[tid] += sdata[tid+i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    sdata[0] = std::pow(sdata[0], (1.0/normType_)); 
  }
  __syncthreads();
  // now we renormalize the blocks that need it
  if (sdata[0] > ScalarConvert<Dtype, Acctype>::to(maxNorm)) {
    Dtype factor = ScalarConvert<Acctype, Dtype>::to(maxNorm / (sdata[0] + 1e-7));
    weight_ptr[idx] *= factor;
    if (tid + blockDim.x < dim)
      weight_ptr[idx + blockDim.x] *= factor;
  }

}

#include "generic/LookupTable.cu"
#include "THCGenerateFloatTypes.h"

