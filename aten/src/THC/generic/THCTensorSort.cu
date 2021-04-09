#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorSort.cu"
#else

#include <c10/cuda/CUDAException.h>

// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
void THCTensor_(sortKeyValueInplace)(THCState* state,
                                     THCTensor* key,
                                     THCudaLongTensor* value,
                                     int dim, bool dir) {
  THArgCheck(key->sizes().equals(value->sizes()), 2,
             "Key tensor must have same size as value tensor");
  int dims = THCudaLongTensor_nDimensionLegacyNoScalars(state, value);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimensionLegacyNoScalars)(state, key);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);

  ptrdiff_t inElements = THCTensor_(nElement)(state, key);

  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = THCTensor_(sizeLegacyNoScalars)(state, key, dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

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
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1,                    \
          GTComp<scalar_t, true>, TYPE, SIZE>                           \
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(        \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          GTComp<scalar_t, true>());                                    \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
    } else {                                                            \
      bitonicSortKVInPlace<scalar_t, int64_t, A, -1,                    \
      LTComp<scalar_t, true>, TYPE, SIZE>                               \
        <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(        \
          keyInfo,                                                      \
          keySlices,                                                    \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          LTComp<scalar_t, true>());                                    \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
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
      TORCH_INTERNAL_ASSERT(false);                                    \
    }                                                   \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  if (THCTensor_canUse32BitIndexMath(state, key)) {
    TensorInfo<scalar_t, unsigned int> keyInfo =
      getTensorInfo<scalar_t, THCTensor, unsigned int>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, unsigned int> valueInfo =
      getTensorInfo<int64_t, THCudaLongTensor, unsigned int>(state, value);
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
    TensorInfo<scalar_t, uint64_t> keyInfo =
      getTensorInfo<scalar_t, THCTensor, uint64_t>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<int64_t, uint64_t> valueInfo =
      getTensorInfo<int64_t, THCudaLongTensor, uint64_t>(state, value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    // int64_t case is rare, just instantiate the generic version
    HANDLE_SORT_CASE(uint64_t, -1);
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE
}

void THCTensor_(sort)(THCState* state,
                      THCTensor *sorted,
                      THCudaLongTensor *indices,
                      THCTensor *input,
                      int dim, int order) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, sorted, input));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, indices));
  dim = at::maybe_wrap_dim(dim, input);
  int64_t dims = THCTensor_(nDimensionLegacyNoScalars)(state, sorted);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);
  dims = THCTensor_(nDimensionLegacyNoScalars)(state, input);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 4, CUTORCH_DIM_WARNING);
  dims = THCudaLongTensor_nDimensionLegacyNoScalars(state, indices);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 3, CUTORCH_DIM_WARNING);

  // Make sure sufficient output space is allocated
  THCTensor_(resizeAs)(state, sorted, input);
  THCudaLongTensor_resize(state, indices, input->sizes(), {});

  // How large are the slices that we are sorting?
  int64_t sliceSize = THCTensor_(sizeLegacyNoScalars)(state, input, dim);

  // Workaround:
  // CUDA 8 uses more shared memory than 7.5 for bitonicSortKVInPlace,
  // and so for the double word types,
  // we get "too many resources requested for launch" in the 2048 case
#if defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_LONG)
  int maxSliceSize = 1024;
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
    TORCH_INTERNAL_ASSERT(false);
  }

  THCudaCheck(cudaGetLastError());
}

#endif
