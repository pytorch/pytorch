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
const int MODE_SUM = 0;
const int MODE_MEAN = 1;

template <typename Dtype, typename Acctype>
__global__ void cunn_LookupTableBag_updateOutputKernel(
  int64_t *input, int64_t *offsets, Dtype *weight, Dtype *output,
  int64_t *offset2bag, int64_t numIndices, int64_t numBags, int64_t stride, int mode,
  int64_t *bag_size) {

  // the strategy here is that each bag x feature is handled by a single thread

  int64_t chunksPerBag = THCCeilDiv(stride, (int64_t) blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < stride) {
      int64_t bag = chunk / chunksPerBag;
      Dtype*  weightFeat = weight + featureDim;
      int64_t begin = offsets[bag] - TH_INDEX_BASE;
      int64_t end = (bag < numBags - 1) ? (offsets[bag + 1] - TH_INDEX_BASE) : numIndices;
      assert(end >= begin);
      Acctype weightFeatSum = ScalarConvert<float, Acctype>::to(0);
      int64_t bag_size_ = 0;
      for (int64_t emb = begin; emb < end; emb++) {
        const int weightRow = ((int) input[emb] - TH_INDEX_BASE) * stride;
        weightFeatSum += ScalarConvert<Dtype, Acctype>::to(weightFeat[weightRow]);
	bag_size_ ++;
        if (featureDim == 0) {
          offset2bag[emb] = bag + TH_INDEX_BASE;
        }
      }
      if (mode == MODE_MEAN) {
	weightFeatSum = weightFeatSum / ScalarConvert<int64_t, Acctype>::to(bag_size_);
	bag_size[bag] = bag_size_;
      }
      (void) MODE_SUM; //silence warnings about unused MODE_SUM;
      output[bag * stride + featureDim] = ScalarConvert<Acctype, Dtype>::to(weightFeatSum);
    }
  }
}

// FIXME: removed the accGradParametersKernelByFeature case present in
// LookupTable. That kernel is faster at small sizes (<768 indices), which
// does not need LookupTableBag (LookupTable + Sum works fine), but would
// still be nice to not be slow in that case.

template <typename Dtype, typename Acctype>
__global__ void cunn_LookupTableBag_accGradParametersKernel(
  int64_t *input, int64_t *indices, Dtype *gradOutput, Dtype *gradWeight, int64_t *offset2bag,
  int64_t *count, Dtype defaultScale, ptrdiff_t numel, int64_t stride,
  int mode, int64_t *bag_size) {

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
      const int seq_number = offset2bag[origRow] - TH_INDEX_BASE;
      const int gradOutputRow = ((int) seq_number) * stride;

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
	  if (mode == MODE_MEAN) {
	    gradient[ii] /= bag_size[seq_number];
	  }
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


#include "generic/LookupTableBag.cu"
#include "THCGenerateFloatTypes.h"
