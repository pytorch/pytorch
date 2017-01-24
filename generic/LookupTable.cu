#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LookupTable.cu"
#else

void THNN_(LookupTable_accGradParameters)(
           THCState *state,
           THCIndexTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCIndexTensor *count,
           THCIndexTensor *sorted,
           THCIndexTensor *indices,
           bool scaleGradByFreq,
           int paddingValue,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, sorted, indices);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  if (!(THCIndexTensor_(isContiguous)(state, input) &&
        THCTensor_(isContiguous)(state, gradWeight)))
  {
    THError("Tensors must be contiguous");
  }

  int nDim = THCIndexTensor_(nDimension)(state, input);
  if (THCIndexTensor_(nDimension)(state, input) != 1 && THCIndexTensor_(nDimension)(state, input) != 2) {
    THCDescBuff s1 = THCIndexTensor_(sizeDesc)(state, input);
    THError("input must be a vector or matrix, but is of shape: %s", s1.str);
  }

  ptrdiff_t numel = THCIndexTensor_(nElement)(state, input);
  long stride = gradWeight->stride[0];

  cudaStream_t stream = THCState_getCurrentStream(state);

  if (numel <= 768 && !scaleGradByFreq) {
    cunn_LookupTable_accGradParametersKernelByFeature<<<DIVUP(stride,4), 128, 0, stream>>>(
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
  THCIndexTensor_(resize)(state, sorted, inputSize, NULL);
  THCIndexTensor_(resize)(state, indices, inputSize, NULL);
  THLongStorage_free(inputSize);

  // Sort the inputs into sorted with the corresponding indices
  THCIndexTensor_(sort)(state, sorted, indices, input, 0, 0);

  THCIndex_t *sorted_data = THCIndexTensor_(data)(state, sorted);
  THCIndex_t  *indices_data = THCIndexTensor_(data)(state, indices);
  THCIndex_t *count_data = NULL;

  if (scaleGradByFreq)
  {
    THCIndexTensor_(resizeAs)(state, count, input);
    count_data = THCIndexTensor_(data)(state, count);

    THCThrustAllocator thrustAlloc(state);
    thrust::device_ptr<THCIndex_t> sorted_ptr(sorted_data);
    thrust::device_ptr<THCIndex_t> count_ptr(count_data);

    // Compute an increasing sequence per unique item in sorted:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 1 2 3 1 2 1 1 2
    thrust::inclusive_scan_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
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
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      thrust::make_reverse_iterator(sorted_ptr + numel),
      thrust::make_reverse_iterator(sorted_ptr),
      thrust::make_reverse_iterator(count_ptr + numel),
      thrust::make_reverse_iterator(count_ptr + numel),
      thrust::equal_to<long>(),
      thrust::maximum<long>()
    );
  }

  dim3 grid(DIVUP(numel,4), DIVUP(stride,128));
  dim3 block(32, 4);
  cunn_LookupTable_accGradParametersKernel<real, accreal><<<grid, block, 0, stream>>>(
    sorted_data,
    indices_data,
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
        THCTensor_(isContiguous)(state, weight)))
  {
    THError("Tensors must be contiguous");
  }
  if (THCIndexTensor_(nDimension)(state, idx) != 1)
    THError("idx must be a vector");
  if (normType <= 0)
    THError("non-positive-norm not supported");

  THCIndex_t numel = THCIndexTensor_(nElement)(state, idx);
  long stride = weight->stride[0];

  // get the unique indices
  thrust::device_ptr<real> weight_ptr(THCTensor_(data)(state, weight));
  thrust::device_ptr<THCIndex_t> idx_ptr(THCIndexTensor_(data)(state, idx));
  thrust::device_ptr<THCIndex_t> end_ptr = thrust::unique(idx_ptr, idx_ptr+numel);
  numel = end_ptr - idx_ptr;

  pow_v<real, accreal> unary_pow(normType);
  thrust::plus<accreal> binary_plus;
  // numel << stride, since idx usually contains sparse row indices
  for (THCIndex_t i = 0; i < numel; i++)
  {
    THCIndex_t k = idx_ptr[i] - TH_INDEX_BASE;
    thrust::device_ptr<real> row_ptr = weight_ptr + k * stride;
    accreal norm = thrust::transform_reduce(row_ptr, row_ptr + stride,
      unary_pow, 0, binary_plus);
    norm = std::pow(norm, (accreal) (1.0 / normType));
    if (norm > ScalarConvert<real, accreal>::to(maxNorm))
    {
      multiply_s<real> unary_mul(ScalarConvert<accreal, real>::to(maxNorm / (norm + 1e-7)));
      thrust::transform(row_ptr, row_ptr + stride, row_ptr, unary_mul);
    }
  }
}

#endif
