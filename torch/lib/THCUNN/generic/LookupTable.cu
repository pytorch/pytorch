#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LookupTable.cu"
#else

void THNN_(LookupTable_accGradParameters)(
           THCState *state,
           THCIndexTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCIndexTensor *count,
           THCIndexTensor *sortedIndices,
           THCIndexTensor *origIndices,
           bool scaleGradByFreq,
           int paddingValue,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, sortedIndices, origIndices);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  if (!(THCIndexTensor_(isContiguous)(state, input) &&
        THCTensor_(isContiguous)(state, gradWeight))) {
    THError("Tensors must be contiguous");
  }

  int nDim = THCIndexTensor_(nDimension)(state, input);
  if (THCIndexTensor_(nDimension)(state, input) != 1 && THCIndexTensor_(nDimension)(state, input) != 2) {
    THCDescBuff s1 = THCIndexTensor_(sizeDesc)(state, input);
    THError("input must be a vector or matrix, but is of shape: %s", s1.str);
  }

  ptrdiff_t numel = THCIndexTensor_(nElement)(state, input);
  long stride = THCTensor_(stride)(state, gradWeight, 0);

  cudaStream_t stream = THCState_getCurrentStream(state);

  if (numel <= 768 && !scaleGradByFreq) {
    dim3 grid(THCCeilDiv(stride, (long) 4));
    dim3 block(128);

    cunn_LookupTable_accGradParametersKernelByFeature<<<grid, block, 0, stream>>>(
      THCIndexTensor_(data)(state, input),
      THCTensor_(data)(state, gradOutput),
      THCTensor_(data)(state, gradWeight),
      scale,
      numel,
      stride,
      paddingValue);
    THCTensor_(free)(state, gradOutput);
    THCudaCheck(cudaGetLastError());
    return;
  }

  THLongStorage *inputSize = THCIndexTensor_(newSizeOf)(state, input);
  THCIndexTensor_(resize)(state, sortedIndices, inputSize, NULL);
  THCIndexTensor_(resize)(state, origIndices, inputSize, NULL);
  THLongStorage_free(inputSize);

  // Sort the inputs into sorted with the corresponding indices; we
  // don't need a stable or multidimensional sort, so just use Thrust
  // directly
  {
    THCIndexTensor_(copy)(state, sortedIndices, input);

    THCThrustAllocator thrustAlloc(state);

    thrust::device_ptr<THCIndex_t>
      sortedIndicesIter(THCIndexTensor_(data)(state, sortedIndices));
    thrust::device_ptr<THCIndex_t>
      origIndicesIter(THCIndexTensor_(data)(state, origIndices));

    // Fill sortedOrigIndices with sequential indices
    thrust::counting_iterator<THCIndex_t> countIter(TH_INDEX_BASE);

    thrust::copy(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      countIter, countIter + numel, origIndicesIter);

    // Sort; a stable sort is not required
    thrust::sort_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      sortedIndicesIter, sortedIndicesIter + numel,
      origIndicesIter, ThrustLTOp<long>());
  }

  THCIndex_t *sortedIndices_data = THCIndexTensor_(data)(state, sortedIndices);
  THCIndex_t *origIndices_data = THCIndexTensor_(data)(state, origIndices);
  THCIndex_t *count_data = NULL;

  if (scaleGradByFreq) {
    THCIndexTensor_(resizeAs)(state, count, input);
    count_data = THCIndexTensor_(data)(state, count);

    THCThrustAllocator thrustAlloc(state);
    thrust::device_ptr<THCIndex_t> sortedIndices_ptr(sortedIndices_data);
    thrust::device_ptr<THCIndex_t> count_ptr(count_data);

    // Compute an increasing sequence per unique item in sortedIndices:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 1 2 3 1 2 1 1 2
    thrust::inclusive_scan_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      sortedIndices_ptr,
      sortedIndices_ptr + numel,
      thrust::make_constant_iterator(1),
      count_ptr
    );

    // Take the maximum of each count per unique key in reverse:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    thrust::inclusive_scan_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      thrust::make_reverse_iterator(sortedIndices_ptr + numel),
      thrust::make_reverse_iterator(sortedIndices_ptr),
      thrust::make_reverse_iterator(count_ptr + numel),
      thrust::make_reverse_iterator(count_ptr + numel),
      thrust::equal_to<long>(),
      thrust::maximum<long>()
    );
  }

  dim3 grid(THCCeilDiv(numel, (ptrdiff_t) 4), THCCeilDiv(stride, (long) 128));
  dim3 block(32, 4);
  cunn_LookupTable_accGradParametersKernel<real, accreal><<<grid, block, 0, stream>>>(
    sortedIndices_data,
    origIndices_data,
    THCTensor_(data)(state, gradOutput),
    THCTensor_(data)(state, gradWeight),
    count_data,
    scale,
    numel,
    stride,
    paddingValue
  );

  THCTensor_(free)(state, gradOutput);
  THCudaCheck(cudaGetLastError());
}

void THNN_(LookupTable_renorm)(
           THCState *state,
           THCIndexTensor *idx,
           THCTensor *weight,
           accreal maxNorm_,
           accreal normType_)
{
  real maxNorm = ScalarConvert<accreal, real>::to(maxNorm_);
  real normType = ScalarConvert<accreal, real>::to(normType_);
  THCUNN_assertSameGPU(state, 2, idx, weight);
  if (!(THCIndexTensor_(isContiguous)(state, idx) &&
        THCTensor_(isContiguous)(state, weight))) {
    THError("Tensors must be contiguous");
  }

  if (THCIndexTensor_(nDimension)(state, idx) != 1) {
    THError("idx must be a vector");
  }

  if (normType <= 0) {
    THError("non-positive-norm not supported");
  }

  THCIndex_t numel = THCIndexTensor_(nElement)(state, idx);
  long stride = THCTensor_(stride)(state, weight, 0);

  // get the unique indices
  thrust::device_ptr<real> weight_ptr(THCTensor_(data)(state, weight));
  thrust::device_ptr<THCIndex_t> idx_ptr(THCIndexTensor_(data)(state, idx));
  thrust::device_ptr<THCIndex_t> end_ptr(thrust::unique(idx_ptr, idx_ptr+numel));
  numel = end_ptr - idx_ptr;

  pow_v<real, accreal> unary_pow(normType);
  thrust::plus<accreal> binary_plus;
  // numel << stride, since idx usually contains sparse row indices
  for (THCIndex_t i = 0; i < numel; i++)
  {
    THCIndex_t k = idx_ptr[i] - TH_INDEX_BASE;
    thrust::device_ptr<real> row_ptr = weight_ptr + k * stride;
    accreal norm = thrust::transform_reduce(row_ptr, row_ptr + stride,
      unary_pow, (accreal)0, binary_plus);
    norm = std::pow(norm, (accreal) (1.0 / normType));
    if (norm > ScalarConvert<real, accreal>::to(maxNorm))
    {
      multiply_s<real> unary_mul(ScalarConvert<accreal, real>::to(maxNorm / (norm + 1e-7)));
      thrust::transform(row_ptr, row_ptr + stride, row_ptr, unary_mul);
    }
  }
}

#endif
