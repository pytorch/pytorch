#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LookupTableSum.cu"
#else


void THNN_(LookupTableSum_updateOutput)(
           THCState *state,
           THCIndexTensor *input,
           THCIndexTensor *offsets,
           THCTensor *weight,
           THCTensor *output,
           THCIndexTensor *offset2bag)
{
  THCUNN_assertSameGPU(state, 5, input, offsets, weight, output, offset2bag);

  if (!(THCIndexTensor_(isContiguous)(state, input) &&
        THCIndexTensor_(isContiguous)(state, offsets) &&
        THCTensor_(isContiguous)(state, weight))) {
    THError("Tensors must be contiguous");
  }

  ptrdiff_t numIndices = THCIndexTensor_(size)(state, input, 0);
  ptrdiff_t numBags = THCIndexTensor_(size)(state, offsets, 0);
  ptrdiff_t stride = THCTensor_(size)(state, weight, 1);

  cudaStream_t stream = THCState_getCurrentStream(state);

  THLongStorage *inputSize = THCIndexTensor_(newSizeOf)(state, input);
  THLongStorage *outputSize = THLongStorage_newWithSize(2);
  outputSize->data[0] = numBags;
  outputSize->data[1] = stride;
  THCTensor_(resize)(state, output, outputSize, NULL);
  THCTensor_(zero)(state, output);
  THCIndexTensor_(resize)(state, offset2bag, inputSize, NULL);
  THLongStorage_free(inputSize);
  THLongStorage_free(outputSize);

  dim3 block = dim3(32, 8);
  int grid = 1024;
  cunn_LookupTableSum_updateOutputKernel<real, accreal><<<grid, block, 0, stream>>>(
    THCIndexTensor_(data)(state, input),
    THCIndexTensor_(data)(state, offsets),
    THCTensor_(data)(state, weight),
    THCTensor_(data)(state, output),
    THCIndexTensor_(data)(state, offset2bag),
    numIndices,
    numBags,
    stride
  );

  THCudaCheck(cudaGetLastError());
}


void THNN_(LookupTableSum_accGradParameters)(
           THCState *state,
           THCIndexTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCIndexTensor *offset2bag,
           THCIndexTensor *count,
           THCIndexTensor *sortedIndices,
           THCIndexTensor *origIndices,
           bool scaleGradByFreq,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCUNN_assertSameGPU(state, 6, input, gradOutput, gradWeight, offset2bag, sortedIndices, origIndices);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);
  if (!(THCIndexTensor_(isContiguous)(state, input) &&
        THCTensor_(isContiguous)(state, gradWeight) &&
        THCIndexTensor_(isContiguous)(state, offset2bag))) {
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
  THCIndex_t *offset2bag_data = THCIndexTensor_(data)(state, offset2bag);
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
  cunn_LookupTableSum_accGradParametersKernel<real, accreal><<<grid, block, 0, stream>>>(
    sortedIndices_data,
    origIndices_data,
    THCTensor_(data)(state, gradOutput),
    THCTensor_(data)(state, gradWeight),
    offset2bag_data,
    count_data,
    scale,
    numel,
    stride
  );

  THCTensor_(free)(state, gradOutput);
  THCudaCheck(cudaGetLastError());
}

#endif
