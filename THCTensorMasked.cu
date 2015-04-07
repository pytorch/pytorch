#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

struct TensorMaskedFillOp {
  TensorMaskedFillOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* t, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    const float maskVal = *mask;
    if (maskVal != 0.0f) {
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

  if (!THCudaTensor_pointwiseApply2(state, tensor, mask, TensorMaskedFillOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCudaTensor_maskedCopy(THCState* state,
                             THCudaTensor *tensor, THCudaTensor *mask, THCudaTensor *src)
{
  THError("maskedCopy is not yet implemented for CUDA");
  THAssert(THCudaTensor_checkGPU(state, 3, tensor, mask, src));
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

  // Determine our output size
  THCudaTensor* contigMask = THCudaTensor_newContiguous(state, mask);
  int totalElements = (int) THCudaTensor_sumall(state, contigMask);
  THCudaTensor_resize1d(state, tensor, totalElements);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaTensor* maskPrefixSum = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, maskPrefixSum, mask);

  thrust::device_ptr<float>
    maskData(THCudaTensor_data(state, contigMask));
  thrust::device_ptr<float>
    maskPrefixSumData(THCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(maskData,
                         maskData + THCudaTensor_nElement(state, contigMask),
                         maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THCudaTensor_pointwiseApply3(
    state, contigMask, maskPrefixSum,
    src, TensorMaskedSelectOp(THCudaTensor_data(state, tensor)));

  THCudaTensor_free(state, contigMask);
  THCudaTensor_free(state, maskPrefixSum);

  if (!status) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

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
  THError("maskedCopyByte is not yet implemented for CUDA");
  THAssert(THCudaTensor_checkGPU(state, 1, tensor));
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
