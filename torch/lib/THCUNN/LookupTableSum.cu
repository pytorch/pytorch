#include "THCUNN.h"
#include "common.h"

#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <thrust/unique.h>
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCTensorSort.cuh"

const int WARP_SIZE = 32;

template <typename Dtype, typename Acctype>
__global__ void cunn_LookupTableSum_updateOutputKernel(
  long *input, long *offsets, Dtype *weight, Dtype *output,
  long *offset2bag, long numIndices, long numBags, long stride) {

  // the strategy here is that each bag x feature is handled by a single thread

  long chunksPerBag = THCCeilDiv(stride, (long) blockDim.x);
  long numChunks = numBags * chunksPerBag;
  long chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  long chunkStride = gridDim.x * blockDim.y;

  for (long chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    long featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < stride) {
      long bag = chunk / chunksPerBag;
      auto weightFeat = weight + featureDim;
      auto begin = offsets[bag] - TH_INDEX_BASE;
      auto end = (bag < numBags - 1) ? (offsets[bag + 1] - TH_INDEX_BASE) : numIndices;
      assert(end >= begin);
      Acctype weightFeatSum = ScalarConvert<float, Acctype>::to(0);
      for (long emb = begin; emb < end; emb++) {
        const int weightRow = ((int) input[emb] - TH_INDEX_BASE) * stride;
        weightFeatSum += ScalarConvert<Dtype, Acctype>::to(weightFeat[weightRow]);
        if (featureDim == 0) {
          offset2bag[emb] = bag + TH_INDEX_BASE;
        }
      }
      output[bag * stride + featureDim] = ScalarConvert<Acctype, Dtype>::to(weightFeatSum);
    }
  }
}

// FIXME: removed the accGradParametersKernelByFeature case present in
// LookupTable. That kernel is faster at small sizes (<768 indices), which
// does not need LookupTableSum (LookupTable + Sum works fine), but would
// still be nice to not be slow in that case.

template <typename Dtype, typename Acctype>
__global__ void cunn_LookupTableSum_accGradParametersKernel(
  long *input, long *indices, Dtype *gradOutput, Dtype *gradWeight, long *offset2bag,
  long *count, Dtype defaultScale, ptrdiff_t numel, long stride) {

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
      && (idx == 0 || input[idx] != input[idx - 1])) {
    do {
      const int startFeature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weightRow = ((int) input[idx] - TH_INDEX_BASE) * stride;

      // Note: only this line changes from LookupTable_accgradParametersKernel
      const int origRow = ((int) indices[idx] - TH_INDEX_BASE);
      const int gradOutputRow = ((int) offset2bag[origRow] - TH_INDEX_BASE) * stride;

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


#include "generic/LookupTableSum.cu"
#include "THCGenerateFloatTypes.h"
