#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorSort.cu"
#else

// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
THC_API void THCTensor_(sortKeyValueInplace)(THCState* state,
                                           THCTensor* key,
                                           THCudaLongTensor* value,
                                           int dim, bool dir) {
  THLongStorage *valueSize = THCudaLongTensor_newSizeOf(state, value);
  THArgCheck(THCTensor_(isSize)(state, key, valueSize), 2,
             "Key tensor must have same size as value tensor");
  THLongStorage_free(valueSize);
  int dims = THCudaLongTensor_nDimension(state, value);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, key);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);

  ptrdiff_t inElements = THCTensor_(nElement)(state, key);
  int64_t keySliceSize = THCTensor_(size)(state, key, dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

  if (THCTensor_(nDimension)(state, key) == 0) {
    // Zero-dim tensor; do nothing
    return;
  }

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (key, value) sort by slice segment
  if (ceilPowerOf2 > 2048) {
    THError("sortKeyValueInplace only works for sizes <= 2048 at present");
  }

  // The grid is based on the number of independent slices that we
  // have to sort; one block per slice
  dim3 grid;
  if (!THC_getGridFromTiles(keySlices, grid)) {
    THError("Slice to sort is too large");
  }

#define HANDLE_CASE(TYPE, A, SIZE)                                      \
  do {                                                                  \
    int blockSize = SIZE / 2;                                           \
    if (blockSize < 1) {                                                \
      blockSize = 1;                                                    \
    }                                                                   \
                                                                        \
    dim3 block(blockSize);                                              \
                                                                        \
    if (dir) {                                                          \
      bitonicSortKVInPlace<real, int64_t, A, -1, GTComp<real>, TYPE, SIZE> \
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(         \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          GTComp<real>());                                              \
    } else {                                                            \
      bitonicSortKVInPlace<real, int64_t, A, -1, LTComp<real>, TYPE, SIZE> \
        <<<grid, block, 0, THCState_getCurrentStream(state)>>>(         \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          LTComp<real>());                                              \
    }                                                                   \
  } while (0)

#define HANDLE_SORT_CASE(TYPE, A)                       \
  {                                                     \
    switch (ceilPowerOf2) {                             \
      case 2048:                                        \
      HANDLE_CASE(TYPE, A, 2048);                       \
      break;                                            \
      case 1024:                                        \
      case 512:                                         \
      case 256:                                         \
      HANDLE_CASE(TYPE, A, 1024);                       \
      break;                                            \
      case 128:                                         \
      case 64:                                          \
      HANDLE_CASE(TYPE, A, 128);                        \
      break;                                            \
      case 32:                                          \
      case 16:                                          \
      case 8:                                           \
      case 4:                                           \
      case 2:                                           \
      HANDLE_CASE(TYPE, A, 32);                         \
      break;                                            \
      case 1:                                           \
      /* Nothing to do, data already sorted */          \
      break;                                            \
      default:                                          \
      assert(false);                                    \
    }                                                   \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  if (TensorUtils<THCTensor>::canUse32BitIndexMath(state, key)) {
    TensorInfo<real, unsigned int> keyInfo =
      getTensorInfo<THCTensor, unsigned int>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, unsigned int> valueInfo =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned int, -2);
    } else {
      switch (keyInfo.dims) {
        case 2:
          HANDLE_SORT_CASE(unsigned int, 2);
          break;
        default:
          HANDLE_SORT_CASE(unsigned int, -1);
          break;
      }
    }
  } else {
    TensorInfo<real, uint64_t> keyInfo =
      getTensorInfo<THCTensor, uint64_t>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, uint64_t> valueInfo =
      getTensorInfo<THCudaLongTensor, uint64_t>(state, value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    // int64_t case is rare, just instantiate the generic version
    HANDLE_SORT_CASE(uint64_t, -1);
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE

  THCudaCheck(cudaGetLastError());
}

void sortViaThrust(THCState* state,
                   THCTensor* sorted,
                   THCudaLongTensor* indices,
                   THCTensor* input,
                   int dim, bool dir) {
  int nDims = THCTensor_(nDimension)(state, input);

  ptrdiff_t totalElements = THCTensor_(nElement)(state, input);
  int64_t sliceSize = THCTensor_(size)(state, input, dim);
  int64_t sliceStride = THCTensor_(stride)(state, input, dim);

  // We perform a vectorized segmented sort in Thrust.
  // Say we are sorting a (2, 3) tensor. We have in flattened form:
  // values 0.4 1.2 5.3 6.2 1.3 2.3
  // indices  0   1   2   3   4   5
  // where indices is a global index (across all slices)

  // First we sort by values, globally:
  // values 6.2 5.3 2.3 1.2 1.3 0.4
  // indices  3   2   5   1   4   0

  // Then we stable sort by segment, which is index / 3:
  // values 5.3 1.2 0.4 6.2 2.3 1.3
  // indices  2   1   0   3   5   4

  // Then we translate the global index to a per-slice Lua index
  // (index % 3) + 1:
  // values 5.3 1.2 0.4 6.2 2.3 1.3
  // indices  3   2   1   1   3   2

  // This method can only work if the slice we are sorting (`dim`) is
  // innermost, and both values and indices are contiguous. We do this
  // by re-arranging the input into this form as needed, which will
  // unfortunately allocate memory if the request is not in this form.
  // Vectorized sort is slower than iterated sort if the number of
  // slices is small (since we're sorting twice, instead of invoking a
  // smaller sort `numSlices` times), but the Thrust sort
  // implementation here is a catch-all, so we're not looking for
  // efficiency, but instead correctness.
  THCTensor_(copy)(state, sorted, input);
  THCTensor* trKeys = THCTensor_(newWithTensor)(state, sorted);
  THCudaLongTensor* trIndices = THCudaLongTensor_newWithTensor(state, indices);

  // Transpose dim to innermost
  if (dim != nDims - 1) {
    THCTensor_(transpose)(state, trKeys, NULL, dim, nDims - 1);
    THCudaLongTensor_transpose(state, trIndices, NULL, dim, nDims - 1);
  }

  // Thrust must operate on a contiguous layout
  THCTensor* trContigKey = THCTensor_(newContiguous)(state, trKeys);
  THCudaLongTensor* trContigIndices = THCudaLongTensor_newContiguous(state, trIndices);

  THCTensor_(free)(state, trKeys);
  THCudaLongTensor_free(state, trIndices);

  THCThrustAllocator thrustAlloc(state);

  thrust::device_ptr<real> keyIter(THCTensor_(data)(state, trContigKey));

  // Since we are composing a global index across all segments rather
  // than a per-segment index, we treat the memory as int so we don't
  // have problems sorting slices < 2^24 but where the entire tensor
  // has more than 2^24 elements
  thrust::device_ptr<int64_t>
    indexIter((int64_t*) THCudaLongTensor_data(state, trContigIndices));

  // Fill the indices with a global index across all slices
  thrust::counting_iterator<int64_t> countIter(0);

  thrust::copy(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    countIter, countIter + totalElements, indexIter);

  // First, we sort globally (across all slices) according to key
  // (the values we're sorting)
  if (dir) {
    thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      keyIter, keyIter + totalElements, indexIter, ThrustGTOp<real>());
  } else {
    thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
      keyIter, keyIter + totalElements, indexIter, ThrustLTOp<real>());
  }

  // Then, re-sort according to slice that each index is
  // in. This completes the segment sort in Thrust, since we're
  // stably sorting here, preserving the relative order of values
  // per each slice
  thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    indexIter, indexIter + totalElements, keyIter,
    SliceComp(sliceSize));

  // Translate the global integer 0-based index to a per-slice real
  // Lua index
  thrust::for_each(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    indexIter, indexIter + totalElements,
    GlobalIndexToPerSliceIndex(sliceSize));

  // Reverse the transposition as needed
  if (dim != nDims - 1) {
    THCTensor_(transpose)(state, trContigKey, NULL, dim, nDims - 1);
    THCudaLongTensor_transpose(state, trContigIndices, NULL, dim, nDims - 1);
  }

  // Then copy back to the expected output
  THCTensor_(freeCopyTo)(state, trContigKey, sorted);
  THCudaLongTensor_freeCopyTo(state, trContigIndices, indices);
}

THC_API void THCTensor_(sort)(THCState* state,
                               THCTensor *sorted,
                               THCudaLongTensor *indices,
                               THCTensor *input,
                               int dim, int order) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, sorted, input));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, indices));
  int64_t dims = THCTensor_(nDimension)(state, sorted);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimension)(state, input);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimension(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);

  // Make sure sufficient output space is allocated
  THCTensor_(resizeAs)(state, sorted, input);
  THLongStorage *inputSize = THCTensor_(newSizeOf)(state, input);
  THCudaLongTensor_resize(state, indices, inputSize, NULL);
  THLongStorage_free(inputSize);

  // How large are the slices that we are sorting?
  int64_t sliceSize = THCTensor_(size)(state, input, dim);

  // Workaround:
  // CUDA 8 uses more shared memory than 7.5 for bitonicSortKVInPlace,
  // and so for the double word types,
  // we get "too many resources requested for launch" in the 2048 case
#if CUDA_VERSION >= 8000
#if defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_LONG)
  int maxSliceSize = 1024;
#else
  int maxSliceSize = 2048;
#endif
#else
  int maxSliceSize = 2048;
#endif

  if (sliceSize <= maxSliceSize) {
    // Fill `indices` (the values) with the
    // slice-relative index.
    THCudaLongTensor_fillSliceWithIndex(state, indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    THCTensor_(copy)(state, sorted, input);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    THCTensor_(sortKeyValueInplace)(state, sorted, indices, dim, order);
  } else {
    // Otherwise, fall back upon Thrust, which handles all other cases
    // (potentially slowly, with extra copies/memory allocations)
    sortViaThrust(state, sorted, indices, input, dim, (bool) order);
  }

  THCudaCheck(cudaGetLastError());
}

#endif
