#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMasked.cu"
#else

#include <ATen/NamedTensorUtils.h>


void THCTensor_(maskedFill)(THCState* state,
                            THCTensor *tensor, THCudaByteTensor *mask, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, mask));
  THArgCheck(THCTensor_(nElement)(state, tensor) ==
             THCudaByteTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THC_pointwiseApply2<scalar_t, uint8_t>(state, tensor, mask,
                                          TensorMaskedFillOp<scalar_t, unsigned char>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(maskedFillBool)(THCState* state,
                                THCTensor *tensor, THCudaBoolTensor *mask, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, mask));
  THArgCheck(THCTensor_(nElement)(state, tensor) ==
             THCudaBoolTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THC_pointwiseApply2<scalar_t, bool>(state, tensor, mask,
                                           TensorMaskedFillOp<scalar_t, bool>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(maskedFillByte)(THCState* state,
                                THCTensor *tensor, THByteTensor *mask, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, tensor));
  THCudaByteTensor* maskCuda = THTensor_wrap(mask).cuda().unsafeReleaseTensorImpl();
  THCTensor_(copy)(state, maskCuda, mask);
  THCTensor_(maskedFill)(state, tensor, maskCuda, value);
  THCudaByteTensor_free(state, maskCuda);
}

void THCTensor_(maskedCopy)(THCState* state,
                            THCTensor *tensor, THCudaByteTensor *mask, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, tensor, src, mask));
  ptrdiff_t maskSize = THCudaByteTensor_nElement(state, mask);
  ptrdiff_t tensorSize = THCTensor_(nElement)(state, tensor);
  ptrdiff_t srcSize = THCTensor_(nElement)(state, src);

  // `mask` and `tensor` must have the same number of elements
  THArgCheck(maskSize == tensorSize, 2,
             "mask and tensor must have the same number of elements");

  // Determine our output size
  ptrdiff_t totalElements = THCudaByteTensor_sumall(state, mask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (totalElements > srcSize) {
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  // FIXME: there appears to be a bug in Thrust (CUDA 7.0) for mixed
  // iterator prefix sums? Convert `mask` to the same datatype as what
  // we're accumulating the prefix sum in (int64_t) to get around it
  THCudaLongTensor* maskLong = THCudaLongTensor_new(state);
  at::IntArrayRef maskSizes = mask->sizes();
  THCudaLongTensor_resize(state, maskLong, maskSizes, {});
  THCTensor_(copy)(state, maskLong, mask);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaLongTensor* maskPrefixSum = THCudaLongTensor_new(state);
  THCudaLongTensor_resize(state, maskPrefixSum, maskSizes, {});

  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<int64_t>
    maskData(THCudaLongTensor_data(state, maskLong));
  thrust::device_ptr<int64_t>
    maskPrefixSumData(THCudaLongTensor_data(state, maskPrefixSum));

  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
    maskData,
    maskData + THCudaLongTensor_nElement(state, maskLong),
    maskPrefixSumData);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THCTensor* contigSrc = THCTensor_(newContiguous)(state, src);

  // update `tensor` where `mask` == 1 but pull from `src` at
  // maskPrefixSum
  bool status = THC_pointwiseApply3<scalar_t, uint8_t, int64_t>(
    state, tensor, mask, maskPrefixSum,
    TensorMaskedCopyOp<scalar_t, unsigned char, int64_t>(
      THCTensor_(data)(state, contigSrc)));

  THCTensor_(free)(state, contigSrc);
  THCudaLongTensor_free(state, maskLong);
  THCudaLongTensor_free(state, maskPrefixSum);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

void THCTensor_(maskedCopyBool)(THCState* state,
                                THCTensor *tensor, THCudaBoolTensor *mask, THCTensor *src)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, tensor, src, mask));
  ptrdiff_t maskSize = THCudaBoolTensor_nElement(state, mask);
  ptrdiff_t tensorSize = THCTensor_(nElement)(state, tensor);
  ptrdiff_t srcSize = THCTensor_(nElement)(state, src);

  // `mask` and `tensor` must have the same number of elements
  THArgCheck(maskSize == tensorSize, 2,
             "mask and tensor must have the same number of elements");

  // Determine our output size
  ptrdiff_t totalElements = THCudaBoolTensor_sumall(state, mask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (totalElements > srcSize) {
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  // FIXME: there appears to be a bug in Thrust (CUDA 7.0) for mixed
  // iterator prefix sums? Convert `mask` to the same datatype as what
  // we're accumulating the prefix sum in (int64_t) to get around it
  THCudaLongTensor* maskLong = THCudaLongTensor_new(state);
  at::IntArrayRef maskSizes = mask->sizes();
  THCudaLongTensor_resize(state, maskLong, maskSizes, {});
  THCTensor_(copy)(state, maskLong, mask);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaLongTensor* maskPrefixSum = THCudaLongTensor_new(state);
  THCudaLongTensor_resize(state, maskPrefixSum, maskSizes, {});

  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<int64_t>
    maskData(THCudaLongTensor_data(state, maskLong));
  thrust::device_ptr<int64_t>
    maskPrefixSumData(THCudaLongTensor_data(state, maskPrefixSum));

  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
    maskData,
    maskData + THCudaLongTensor_nElement(state, maskLong),
    maskPrefixSumData);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THCTensor* contigSrc = THCTensor_(newContiguous)(state, src);

  // update `tensor` where `mask` == 1 but pull from `src` at
  // maskPrefixSum
  bool status = THC_pointwiseApply3<scalar_t, bool, int64_t>(
    state, tensor, mask, maskPrefixSum,
    TensorMaskedCopyOp<scalar_t, bool, int64_t>(
      THCTensor_(data)(state, contigSrc)));

  THCTensor_(free)(state, contigSrc);
  THCudaLongTensor_free(state, maskLong);
  THCudaLongTensor_free(state, maskPrefixSum);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

void THCTensor_(maskedCopyByte)(THCState* state,
                                THCTensor *tensor, THByteTensor *mask, THCTensor *src) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCudaByteTensor* maskCuda = THTensor_wrap(mask).cuda().unsafeReleaseTensorImpl();
  THCTensor_(copy)(state, maskCuda, mask);
  THCTensor_(maskedCopy)(state, tensor, maskCuda, src);
  THCudaByteTensor_free(state, maskCuda);
}

void THCTensor_(maskedSelect)(THCState* state,
                              THCTensor* tensor, THCTensor* src, THCudaByteTensor* mask) {
  at::NoNamesGuard guard;
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, tensor, src, mask));
  THArgCheck(THCudaByteTensor_nElement(state, mask) ==
             THCTensor_(nElement)(state, src),
             2, "sizes do not match");

  // Determine our output size
  ptrdiff_t totalElements = THCudaByteTensor_sumall(state, mask);
  THCTensor* tensorContig = THCTensor_(newContiguous)(state, tensor);

  THCTensor_(resize1d)(state, tensorContig, totalElements);
  if (tensor != tensorContig) {
    THCTensor_(resize1d)(state, tensor, totalElements);
  }

  // FIXME: there appears to be a bug in Thrust (CUDA 7.0) for mixed
  // iterator prefix sums? Convert `mask` to the same datatype as what
  // we're accumulating the prefix sum in (int64_t) to get around it
  THCudaLongTensor* maskLong = THCudaLongTensor_new(state);
  at::IntArrayRef maskSizes = mask->sizes();
  THCudaLongTensor_resize(state, maskLong, maskSizes, {});
  THCTensor_(copy)(state, maskLong, mask);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaLongTensor* maskPrefixSum = THCudaLongTensor_new(state);
  THCudaLongTensor_resize(state, maskPrefixSum, maskSizes, {});

  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<int64_t>
    maskData(THCudaLongTensor_data(state, maskLong));
  thrust::device_ptr<int64_t>
    maskPrefixSumData(THCudaLongTensor_data(state, maskPrefixSum));

  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
    maskData,
    maskData + THCudaLongTensor_nElement(state, maskLong),
    maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THC_pointwiseApply3<uint8_t, int64_t, scalar_t>(
    state, mask, maskPrefixSum,
    src, TensorMaskedSelectOp<scalar_t, unsigned char, int64_t>(
      THCTensor_(data)(state, tensor)));

  THCudaLongTensor_free(state, maskLong);
  THCudaLongTensor_free(state, maskPrefixSum);

  if (tensor != tensorContig) {
    THCTensor_(freeCopyTo)(state, tensorContig, tensor);
  } else {
    THCTensor_(free)(state, tensorContig);
  }

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

void THCTensor_(maskedSelectBool)(THCState* state,
                                   THCTensor* tensor, THCTensor* src, THCudaBoolTensor* mask) {
  at::NoNamesGuard guard;
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, tensor, src, mask));
  THArgCheck(THCudaBoolTensor_nElement(state, mask) ==
             THCTensor_(nElement)(state, src),
             2, "sizes do not match");

  // Determine our output size
  ptrdiff_t totalElements = THCudaBoolTensor_sumall(state, mask);
  THCTensor* tensorContig = THCTensor_(newContiguous)(state, tensor);

  THCTensor_(resize1d)(state, tensorContig, totalElements);
  if (tensor != tensorContig) {
    THCTensor_(resize1d)(state, tensor, totalElements);
  }

  // FIXME: there appears to be a bug in Thrust (CUDA 7.0) for mixed
  // iterator prefix sums? Convert `mask` to the same datatype as what
  // we're accumulating the prefix sum in (int64_t) to get around it
  THCudaLongTensor* maskLong = THCudaLongTensor_new(state);
  at::IntArrayRef maskSizes = mask->sizes();
  THCudaLongTensor_resize(state, maskLong, maskSizes, {});
  THCTensor_(copy)(state, maskLong, mask);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaLongTensor* maskPrefixSum = THCudaLongTensor_new(state);
  THCudaLongTensor_resize(state, maskPrefixSum, maskSizes, {});

  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<int64_t>
    maskData(THCudaLongTensor_data(state, maskLong));
  thrust::device_ptr<int64_t>
    maskPrefixSumData(THCudaLongTensor_data(state, maskPrefixSum));

  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
    maskData,
    maskData + THCudaLongTensor_nElement(state, maskLong),
    maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THC_pointwiseApply3<bool, int64_t, scalar_t>(
    state, mask, maskPrefixSum,
    src, TensorMaskedSelectOp<scalar_t, bool, int64_t>(
      THCTensor_(data)(state, tensor)));

  THCudaLongTensor_free(state, maskLong);
  THCudaLongTensor_free(state, maskPrefixSum);

  if (tensor != tensorContig) {
    THCTensor_(freeCopyTo)(state, tensorContig, tensor);
  } else {
    THCTensor_(free)(state, tensorContig);
  }

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THCudaCheck(cudaGetLastError());
}

// FIXME: remove now that we have THCudaByteTensor?
void THCTensor_(maskedSelectByte)(THCState* state,
                                  THCTensor *tensor, THCTensor *src, THByteTensor *mask)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCudaByteTensor* maskCuda = THTensor_wrap(mask).cuda().unsafeReleaseTensorImpl();
  THCTensor_(copy)(state, maskCuda, mask);
  THCTensor_(maskedSelect)(state, tensor, src, maskCuda);
  THCudaByteTensor_free(state, maskCuda);
}

#endif
