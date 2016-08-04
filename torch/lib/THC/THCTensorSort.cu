#include "THCReduceApplyUtils.cuh"
#include "THCSortUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorTypeUtils.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
unsigned long nextHighestPowerOf2(unsigned long n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;

  return n;
}

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename IndexType, int Dim>
__global__ void
fillSliceWithIndex(TensorInfo<float, IndexType> out,
                   IndexType totalSlices,
                   IndexType sliceSize,
                   IndexType sliceStride) {
  IndexType slice = getLinearBlockId<IndexType>();

  if (slice >= totalSlices) {
    return;
  }

  const unsigned long offset =
    IndexToOffset<float, IndexType, Dim>::get(slice, out);
  float* base = &out.data[offset];

  for (long i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = (float) i + 1.0f;
  }
}

void THCudaTensor_fillSliceWithIndex(THCState* state,
                                     THCudaTensor* t,
                                     int dim) {
  THCCheckTensorDims(state, t, 2);

  long inElements = THCudaTensor_nElement(state, t);
  long sliceSize = THCudaTensor_size(state, t, dim);
  long numSlices = inElements / sliceSize;

  dim3 grid;
  if (!THC_getGridFromTiles(numSlices, grid)) {
    THError("Slice to fill with indices is too large");
  }

  long maxThreads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  long numThreads = sliceSize;
  if (numThreads > maxThreads) {
    numThreads = maxThreads;
  }

  dim3 block(numThreads);

#define FILL_INDEX(T, DIM)                                       \
  fillSliceWithIndex<T, DIM>                                     \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(      \
      info, numSlices, sliceSize, info.strides[collapseDim])

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, t)) {
    TensorInfo<float, unsigned int> info =
      getTensorInfo<THCudaTensor, unsigned int>(state, t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    if (info.isContiguous()) {
      FILL_INDEX(unsigned int, -2);
    } else {
      if (info.dims == 1) {
        FILL_INDEX(unsigned int, 1);
      } else if (info.dims == 2) {
        FILL_INDEX(unsigned int, 2);
      } else {
        FILL_INDEX(unsigned int, -1);
      }
    }
  } else {
    TensorInfo<float, unsigned long> info =
      getTensorInfo<THCudaTensor, unsigned long>(state, t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    // catch-all implementation
    FILL_INDEX(unsigned long, -1);
  }

#undef FILL_INDEX

  THCudaCheck(cudaGetLastError());
}

// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
THC_API void THCudaTensor_sortKeyValueInplace(THCState* state,
                                              THCudaTensor* key,
                                              THCudaTensor* value,
                                              int dim, bool dir) {
  THArgCheck(THCudaTensor_isSameSizeAs(state, key, value), 2,
             "Key tensor must have same size as value tensor");
  THCCheckTensorDims(state, key, 2);
  THCCheckTensorDims(state, value, 3);

  long inElements = THCudaTensor_nElement(state, key);
  long keySliceSize = THCudaTensor_size(state, key, dim);
  long keySlices = inElements / keySliceSize;

  if (THCudaTensor_nDimension(state, key) == 0) {
    // Zero-dim tensor; do nothing
    return;
  }

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  long ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (key, value) sort by slice segment
  if (ceilPowerOf2 > 2048) {
    THError("sortKeyValueInplace only works for sizes <= 2048 at present");
  }

  int blockSize = (int) ceilPowerOf2 / 2;
  if (blockSize < 1) {
    blockSize = 1;
  }

  dim3 block(blockSize);

  // The grid is based on the number of independent slices that we
  // have to sort; one block per slice
  dim3 grid;
  if (!THC_getGridFromTiles(keySlices, grid)) {
    THError("Slice to sort is too large");
  }

#define HANDLE_CASE(TYPE, A, SIZE)                                      \
  if (dir) {                                                            \
    bitonicSortKVInPlace<float, float, A, -1, GTComp<float>, TYPE, SIZE> \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(           \
        keyInfo,                                                        \
        keySlices,                                                      \
        (TYPE) keySliceSize,                                            \
        (TYPE) keyInfo.strides[collapseKeyDim],                         \
        valueInfo,                                                      \
        (TYPE) valueInfo.strides[collapseValueDim],                     \
        GTComp<float>());                                               \
  } else {                                                              \
    bitonicSortKVInPlace<float, float, A, -1, LTComp<float>, TYPE, SIZE> \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(           \
        keyInfo,                                                        \
        keySlices,                                                      \
        (TYPE) keySliceSize,                                            \
        (TYPE) keyInfo.strides[collapseKeyDim],                         \
        valueInfo,                                                      \
        (TYPE) valueInfo.strides[collapseValueDim],                     \
        LTComp<float>());                                               \
  }

#define HANDLE_SORT_CASE(TYPE, A)                       \
  {                                                     \
    switch (ceilPowerOf2) {                             \
      case 2048:                                        \
      HANDLE_CASE(TYPE, A, 2048);                       \
      break;                                            \
      case 1024:                                        \
      HANDLE_CASE(TYPE, A, 1024);                       \
      break;                                            \
      case 512:                                         \
      HANDLE_CASE(TYPE, A, 512);                        \
      break;                                            \
      case 256:                                         \
      HANDLE_CASE(TYPE, A, 256);                        \
      break;                                            \
      case 128:                                         \
      HANDLE_CASE(TYPE, A, 128);                        \
      break;                                            \
      case 64:                                          \
      HANDLE_CASE(TYPE, A, 64);                         \
      break;                                            \
      case 32:                                          \
      HANDLE_CASE(TYPE, A, 32);                         \
      break;                                            \
      case 16:                                          \
      HANDLE_CASE(TYPE, A, 16);                         \
      break;                                            \
      case 8:                                           \
      HANDLE_CASE(TYPE, A, 8);                          \
      break;                                            \
      case 4:                                           \
      HANDLE_CASE(TYPE, A, 4);                          \
      break;                                            \
      case 2:                                           \
      HANDLE_CASE(TYPE, A, 2);                          \
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
  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, key)) {
    TensorInfo<float, unsigned int> keyInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<float, unsigned int> valueInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned int, -2);
    } else {
      switch (keyInfo.dims) {
        case 1:
          HANDLE_SORT_CASE(unsigned int, 1);
          break;
        case 2:
          HANDLE_SORT_CASE(unsigned int, 2);
          break;
        default:
          HANDLE_SORT_CASE(unsigned int, -1);
          break;
      }
    }
  } else {
    TensorInfo<float, unsigned long> keyInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, key);
    keyInfo.reduceDim(dim);
    int collapseKeyDim = keyInfo.collapseDims(dim);

    TensorInfo<float, unsigned long> valueInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, value);
    valueInfo.reduceDim(dim);
    int collapseValueDim = valueInfo.collapseDims(dim);

    // long case is rare, just instantiate these versions
    if (keyInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned long, -2);
    } else {
      HANDLE_SORT_CASE(unsigned long, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE

  THCudaCheck(cudaGetLastError());
}

// For slice sorting in Thrust; extracts a slice index from a linear
// index and uses that for comparison
struct SliceComp {
  SliceComp(int size) : sliceSize(size) {}

  __device__ bool operator()(const int& a, const int& b) const {
    // Since the slices are guaranteed to be innermost, the segment is
    // just via integer division
    int segA = a / sliceSize;
    int segB = b / sliceSize;
    return segA < segB;
  }

  const int sliceSize;
};

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  GlobalIndexToPerSliceIndex(int size) : sliceSize(size) {}

  __device__ inline void operator()(int& v) const {
    // Thrust is operating on this index array as an array of type
    // int, but to Torch it should be a float array.
    v = __float_as_int((float) (v % sliceSize + 1));
  }

  const int sliceSize;
};

void sortViaThrust(THCState* state,
                   THCudaTensor* sorted,
                   THCudaTensor* indices,
                   THCudaTensor* input,
                   int dim, bool dir) {
  long nDims = THCudaTensor_nDimension(state, input);

  long totalElements = THCudaTensor_nElement(state, input);
  long sliceSize = THCudaTensor_size(state, input, dim);
  long sliceStride = THCudaTensor_stride(state, input, dim);

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
  THCudaTensor_copy(state, sorted, input);
  THCudaTensor* trKeys = THCudaTensor_newWithTensor(state, sorted);
  THCudaTensor* trIndices = THCudaTensor_newWithTensor(state, indices);

  // Transpose dim to innermost
  if (dim != nDims - 1) {
    THCudaTensor_transpose(state, trKeys, NULL, dim, nDims - 1);
    THCudaTensor_transpose(state, trIndices, NULL, dim, nDims - 1);
  }

  // Thrust must operate on a contiguous layout
  THCudaTensor* trContigKey = THCudaTensor_newContiguous(state, trKeys);
  THCudaTensor* trContigIndices = THCudaTensor_newContiguous(state, trIndices);

  THCudaTensor_free(state, trKeys);
  THCudaTensor_free(state, trIndices);

  thrust::device_ptr<float> keyIter(THCudaTensor_data(state, trContigKey));

  // Since we are composing a global index across all segments rather
  // than a per-segment index, we treat the memory as int so we don't
  // have problems sorting slices < 2^24 but where the entire tensor
  // has more than 2^24 elements
  thrust::device_ptr<int>
    indexIter((int*) THCudaTensor_data(state, trContigIndices));

  // Fill the indices with a global index across all slices
  thrust::counting_iterator<int> countIter(0);

  thrust::copy(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    countIter, countIter + totalElements, indexIter);

  // First, we sort globally (across all slices) according to key
  // (the values we're sorting)
  if (dir) {
    thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      keyIter, keyIter + totalElements, indexIter, thrust::greater<float>());
  } else {
    thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      keyIter, keyIter + totalElements, indexIter, thrust::less<float>());
  }

  // Then, re-sort according to slice that each index is
  // in. This completes the segment sort in Thrust, since we're
  // stably sorting here, preserving the relative order of values
  // per each slice
  thrust::stable_sort_by_key(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    indexIter, indexIter + totalElements, keyIter,
    SliceComp(sliceSize));

  // Translate the global integer 0-based index to a per-slice float
  // Lua index
  thrust::for_each(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    indexIter, indexIter + totalElements,
    GlobalIndexToPerSliceIndex(sliceSize));

  // Reverse the transposition as needed
  if (dim != nDims - 1) {
    THCudaTensor_transpose(state, trContigKey, NULL, dim, nDims - 1);
    THCudaTensor_transpose(state, trContigIndices, NULL, dim, nDims - 1);
  }

  // Then copy back to the expected output
  THCudaTensor_freeCopyTo(state, trContigKey, sorted);
  THCudaTensor_freeCopyTo(state, trContigIndices, indices);
}

THC_API void THCudaTensor_sort(THCState* state,
                               THCudaTensor *sorted,
                               THCudaTensor *indices,
                               THCudaTensor *input,
                               int dim, int order) {
  THAssert(THCudaTensor_checkGPU(state, 3, sorted, indices, input));
  THCCheckTensorDims(state, sorted, 2);
  THCCheckTensorDims(state, indices, 3);
  THCCheckTensorDims(state, input, 4);

  // Make sure sufficient output space is allocated
  THCudaTensor_resizeAs(state, sorted, input);
  THCudaTensor_resizeAs(state, indices, input);

  // How large are the slices that we are sorting?
  long sliceSize = THCudaTensor_size(state, input, dim);

  // We're using THCudaTensor to write out indices, so if the slice
  // size that we're sorting has more elements than can be
  // represented in fp32, warn the user
  // FIXME: this isn't a real restriction of either our code or of
  // Thrust, but we have to switch to a CUDA long tensor to support
  // larger slice sizes. Otherwise the indices will contain garbage.
  THArgCheck(sliceSize <= (long) FLOAT32_MAX_CONSECUTIVE_INT, 5,
             "The sort dimension exceeds single-precision float "
             "consecutive integer precision size (2^24), since float "
             "is used for indices");

  if (sliceSize <= 2048) {
    // Fill `indices` (the values) with the
    // slice-relative index.
    THCudaTensor_fillSliceWithIndex(state, indices, dim);

    // We sort k/v pairs in-place; copy unsorted input to output
    THCudaTensor_copy(state, sorted, input);

    // Sort using our in-place k/v kernel that supports arbitrary
    // layout
    THCudaTensor_sortKeyValueInplace(state, sorted, indices, dim, order);
  } else {
    // Otherwise, fall back upon Thrust, which handles all other cases
    // (potentially slowly, with extra copies/memory allocations)
    sortViaThrust(state, sorted, indices, input, dim, (bool) order);
  }

  THCudaCheck(cudaGetLastError());
}
