#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCDeviceUtils.cuh"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

struct TensorMaskedFillOp {
  TensorMaskedFillOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* t, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      *t = value;
    }
  }

  float value;
};

void THCudaTensor_maskedFill(THCState* state,
                             THCudaTensor *tensor, THCudaTensor *mask, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 2, tensor, mask));
  THArgCheck(THCudaTensor_nElement(state, tensor) ==
             THCudaTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THC_pointwiseApply2(state, tensor, mask, TensorMaskedFillOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

struct TensorMaskedCopyOp {
  TensorMaskedCopyOp(float* s) : src(s) {}

  __device__ __forceinline__ void operator()(float* out, float* mask, float* maskPrefixSum) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      // We've already checked that this offset is <= 2^24, so this is ok.
      *out = src[(int) *maskPrefixSum];
    }
  }

  // Where we are copying from
  float* src;
};


void THCudaTensor_maskedCopy(THCState* state,
                             THCudaTensor *tensor, THCudaTensor *mask, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 3, tensor, src, mask));
  long maskSize = THCudaTensor_nElement(state, mask);
  long tensorSize = THCudaTensor_nElement(state, tensor);
  long srcSize = THCudaTensor_nElement(state, src);

  // Since we are performing a prefix sum of mask, it cannot exceed
  // the size allowed in consecutive integers in float32
  THArgCheck(maskSize <= (long) FLOAT32_MAX_CONSECUTIVE_INT,
             3, "mask nElements exceeds single-precision float "
             "consecutive integer precision size (2^24)");

  // `mask` and `tensor` must have the same number of elements
  THArgCheck(maskSize == tensorSize, 2,
             "mask and tensor must have the same number of elements");

  THCudaTensor* contigMask = THCudaTensor_newContiguous(state, mask);
  long oneElements = (long) THCudaTensor_sumall(state, contigMask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (oneElements > srcSize) {
    THCudaTensor_free(state, contigMask);
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  // Use a prefix sum to determine the copy locations of the masked elements
  THCudaTensor* maskPrefixSum = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, maskPrefixSum, contigMask);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THCudaTensor* contigSrc = THCudaTensor_newContiguous(state, src);

  thrust::device_ptr<float>
    maskData(THCudaTensor_data(state, contigMask));
  thrust::device_ptr<float>
    maskPrefixSumData(THCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
                         maskData,
                         maskData + THCudaTensor_nElement(state, contigMask),
                         maskPrefixSumData);

  // update `tensor` where `mask` == 1 but pull from `src` at
  // maskPrefixSum
  bool status = THC_pointwiseApply3(
    state, tensor, contigMask, maskPrefixSum,
    TensorMaskedCopyOp(THCudaTensor_data(state, contigSrc)));

  THCudaTensor_free(state, contigSrc);
  THCudaTensor_free(state, maskPrefixSum);
  THCudaTensor_free(state, contigMask);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

struct TensorMaskedSelectOp {
  TensorMaskedSelectOp(float* t) : out(t) {}
  __device__ __forceinline__ void operator()(float* mask, float* maskPrefixSum, float* in) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      out[(int) *maskPrefixSum] = *in;
    }
  }

  float* out;
};

void THCudaTensor_maskedSelect(THCState* state,
                               THCudaTensor *tensor, THCudaTensor *src, THCudaTensor *mask)
{
  THAssert(THCudaTensor_checkGPU(state, 3, tensor, src, mask));
  THArgCheck(THCudaTensor_nElement(state, mask) == THCudaTensor_nElement(state, src),
             2, "sizes do not match");

  // Since we are performing a prefix sum of mask, it cannot exceed
  // the size allowed in consecutive integers in float32
  THArgCheck(THCudaTensor_nElement(state, mask) <=
             (long) FLOAT32_MAX_CONSECUTIVE_INT,
             3, "mask nElements exceeds single-precision float "
             "consecutive integer precision size (2^24)");

  // Determine our output size
  THCudaTensor* contigMask = THCudaTensor_newContiguous(state, mask);
  long totalElements = (long) THCudaTensor_sumall(state, contigMask);

  // This should be contiguous already, so no need to make it contig
  // for the apply kernel
  THCudaTensor_resize1d(state, tensor, totalElements);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaTensor* maskPrefixSum = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, maskPrefixSum, contigMask);

  thrust::device_ptr<float>
    maskData(THCudaTensor_data(state, contigMask));
  thrust::device_ptr<float>
    maskPrefixSumData(THCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
                         maskData,
                         maskData + THCudaTensor_nElement(state, contigMask),
                         maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THC_pointwiseApply3(
    state, contigMask, maskPrefixSum,
    src, TensorMaskedSelectOp(THCudaTensor_data(state, tensor)));

  THCudaTensor_free(state, contigMask);
  THCudaTensor_free(state, maskPrefixSum);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_maskedFillByte(THCState* state, THCudaTensor *tensor, THByteTensor *mask, float value)
{
  THAssert(THCudaTensor_checkGPU(state, 1, tensor));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THCudaTensor_maskedFill(state, tensor, maskCuda, value);
  THCudaTensor_free(state, maskCuda);
}

void THCudaTensor_maskedCopyByte(THCState* state, THCudaTensor *tensor, THByteTensor *mask, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, tensor, src));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THCudaTensor_maskedCopy(state, tensor, maskCuda, src);
  THCudaTensor_free(state, maskCuda);
}

void THCudaTensor_maskedSelectByte(THCState* state, THCudaTensor *tensor, THCudaTensor *src, THByteTensor *mask)
{
  THAssert(THCudaTensor_checkGPU(state, 2, tensor, src));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THCudaTensor_maskedSelect(state, tensor, src, maskCuda);
  THCudaTensor_free(state, maskCuda);
}
