#include "THGeneral.h"
#include "THCGeneral.h"
#include "THCTensor.h"
#include <assert.h>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

// backward-compatible LDG
#if __CUDA_ARCH__ >= 350
#define LDG(x) (__ldg(x))
#else
#define LDG(x) (*(x))
#endif

// Maximum elements per thread that we will copy
#define ELEMENTS_PER_THREAD 8L

// Threads per thread block
#define THREADS_PER_BLOCK 32 * 4

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
#define MAX_GRID_SIZE 65535L

// Maximum number of dimensions allowed for cutorch
#define MAX_DIMS 25

template <typename IndexType>
struct TensorInfo {
  float* data;
  IndexType sizes[MAX_DIMS];
  IndexType strides[MAX_DIMS];
  int dims;
};

// This function extracts size/stride information for the kernel.
// Successive dimensions can be collapsed if the size/strides match
// up and thus there are no holes between the dimensions. This is used
// to reduce the complexity of the problem.
template <typename IndexType>
TensorInfo<IndexType>
THCudaTensor_computeTensorInfo(THCudaTensor* t) {
  int dims = THCudaTensor_nDimension(t);
  assert(dims <= MAX_DIMS);

  TensorInfo<IndexType> info;
  info.data = THCudaTensor_data(t);

  // Count the number of successive dimensions that can be collapsed, from
  // innermost to outermost.
  int numCollapsed = 0;

  // Find the innermost dimension not of size 1, since dimensions of size 1 are
  // collapsible.
  int firstNonOneDim = -1;

  for (int i = dims - 1; i >= 0; --i) {
    if (THCudaTensor_size(t, i) != 1) {
      firstNonOneDim = i;
      break;
    }
  }

  // We guarantee that we are never called with only dimensions of size 1.
  assert(firstNonOneDim >= 0);

  // Skip the leading size 1 dims
  numCollapsed += dims - 1 - firstNonOneDim;

  // Now, to determine the other collapsible dims. These are the size/strides
  // of the previous inner non-collapsible dim we encounter.
  long sizeInner = THCudaTensor_size(t, firstNonOneDim);
  long strideInner = THCudaTensor_stride(t, firstNonOneDim);

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = THCudaTensor_size(t, i);
    long strideOuter = THCudaTensor_stride(t, i);

    // The next outermost dimension can be skipped if size 1
    if (sizeOuter == 1) {
      ++numCollapsed;
      continue;
    }

    // If the next outermost dimension is contiguous with the
    // previous non-collapsed one, collapse it
    if (strideOuter == strideInner * sizeInner) {
      ++numCollapsed;

      // This is the run of collapsed dimensions' size
      sizeInner = sizeInner * sizeOuter;
      continue;
    }

    // Otherwise, this new outer dimension at `i` cannot be collapsed
    // and is different from the previous.
    sizeInner = sizeOuter;
    strideInner = strideOuter;
  }

  assert(numCollapsed < dims);
  info.dims = dims - numCollapsed;

  // Determine the sizes of the collapsed dimensions.
  int collapsedIndex = dims - numCollapsed - 1;
  info.sizes[collapsedIndex] = THCudaTensor_size(t, firstNonOneDim);
  info.strides[collapsedIndex] = THCudaTensor_stride(t, firstNonOneDim);

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = THCudaTensor_size(t, i);
    long strideOuter = THCudaTensor_stride(t, i);

    if (sizeOuter == 1) {
      // skip
      continue;
    }

    if (strideOuter ==
        info.sizes[collapsedIndex] * info.strides[collapsedIndex]) {
      // collapse
      info.sizes[collapsedIndex] *= sizeOuter;
      continue;
    }

    // Otherwise, strides don't match; dimension `i` is not collapsible.
    --collapsedIndex;
    assert(collapsedIndex >= 0);
    info.sizes[collapsedIndex] = sizeOuter;
    info.strides[collapsedIndex] = strideOuter;
  }

  // We must have filled all the dimensions we're looking for
  assert(collapsedIndex == 0);

  // Fill out the remainder dims for sanity.
  for (int i = dims - numCollapsed; i < MAX_DIMS; ++i) {
    info.sizes[i] = 1;
    info.strides[i] = info.strides[dims - numCollapsed - 1] *
      info.sizes[dims - numCollapsed - 1];
  }

  return info;
}

// Returns true if all linear ID -> offset math can be performed using 32 bit
// unsigned math
bool
canUse32BitCopyMath(THCudaTensor* t) {
  long elements = THCudaTensor_nElement(t);
  if (elements >= UINT_MAX) {
    return false;
  }

  long offset = 0;
  long linearId = elements - 1;

  for (int i = THCudaTensor_nDimension(t) - 1; i >= 0; --i) {
    long curDimIndex = linearId % THCudaTensor_size(t, i);
    long curDimOffset = curDimIndex * THCudaTensor_stride(t, i);
    offset += curDimOffset;
    linearId /= THCudaTensor_size(t, i);
  }

  if (offset >= UINT_MAX) {
    return false;
  }

  return true;
}

// Translate a linear ID for the copy to a float offset
template <typename IndexType, int Dims>
__forceinline__ __device__ IndexType
linearIdToOffset(IndexType linearId, const TensorInfo<IndexType>& info) {
  IndexType offset = 0;

  if (Dims == -1) {
    // Use dynamic dims
    for (int i = info.dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      linearId /= info.sizes[i];
    }
  } else {
    // Use static dims
    for (int i = Dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      if (i > 0) {
        linearId /= info.sizes[i];
      }
    }
  }

  return offset;
}

// Both `src` and `dst` have the same number of total elements, which are copied
// based on a linear id.
template <typename IndexType, int DstDims, int SrcDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 4, 16)
#endif
__global__ void
THCudaTensor_kernel_copy(TensorInfo<IndexType> dst,
                         TensorInfo<IndexType> src,
                         IndexType totalElements) {
  const IndexType linearBlockId =
    blockIdx.z * gridDim.y * gridDim.x +
    blockIdx.y * gridDim.x +
    blockIdx.x;

  const IndexType startLinearId =
    linearBlockId * THREADS_PER_BLOCK * ELEMENTS_PER_THREAD;

  IndexType endLinearId =
    (linearBlockId + 1) * THREADS_PER_BLOCK * ELEMENTS_PER_THREAD;
  endLinearId = endLinearId < totalElements ? endLinearId : totalElements;

  for (IndexType linearId = startLinearId + threadIdx.x;
       linearId < endLinearId;
       linearId += THREADS_PER_BLOCK) {
    // Convert `linearId` into an offset of `src`
    const IndexType srcOffset =
      linearIdToOffset<IndexType, SrcDims>(linearId, src);

    // Convert `linearId` into an offset of `dst`
    const IndexType dstOffset =
      linearIdToOffset<IndexType, DstDims>(linearId, dst);

    dst.data[dstOffset] = LDG(&src.data[srcOffset]);
  }
}

THC_API void
THCudaTensor_copy(THCudaTensor* dst, THCudaTensor* src) {
  long totalElements = THCudaTensor_nElement(dst);

  THArgCheck(totalElements == THCudaTensor_nElement(src), 2,
             "sizes do not match");

  THArgCheck(THCudaTensor_nDimension(dst) <= MAX_DIMS, 2,
             "Copy only supported for <= 25 dimensions");
  THArgCheck(THCudaTensor_nDimension(src) <= MAX_DIMS, 3,
             "Copy only supported for <= 25 dimensions");

  if (THCudaTensor_nDimension(dst) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }

  // We can memcpy the memory if:
  // -both tensors are contiguous; or,
  // -there is only one element to copy; or,
  // -FIXME: if both tensors have matching size and stride arrays, and no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is contiguous).
  bool memcpyEligible =
    (THCudaTensor_isContiguous(dst) && THCudaTensor_isContiguous(src)) ||
    (totalElements == 1);

  if (memcpyEligible) {
    THCudaCheck(cudaMemcpyAsync(THCudaTensor_data(dst),
                                THCudaTensor_data(src),
                                totalElements * sizeof(float),
                                cudaMemcpyDeviceToDevice));
  } else {
    // We always work with a THREADS_PER_BLOCK-sized thread block,
    // and assume a max sized grid dimension of MAX_GRID_SIZE.
    // Each thread will process up to ELEMENTS_PER_THREAD elements.
    const dim3 block(THREADS_PER_BLOCK);

    long gridTiles = DIVUP(totalElements, block.x * ELEMENTS_PER_THREAD);
    THArgCheck(gridTiles <= MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE, 2,
               "tensor too large");

    long gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    long gridY = 1;
    long gridZ = 1;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = DIVUP(gridTiles, MAX_GRID_SIZE);
      gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

      if (gridTiles > MAX_GRID_SIZE) {
        gridTiles = DIVUP(gridTiles, MAX_GRID_SIZE);
        gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
      }
    }

    dim3 grid(gridX, gridY, gridZ);

    // It is possible that the tensor dimensions are able to be collapsed,
    // and thus we can reduce the actual code complexity of the copy by
    // exploiting this knowledge statically, since the div/mod is the
    // most expensive part of the operation, more so than memory accesses.
    // For instance, when copying a non-contiguous to a contiguous tensor
    // (or vice versa), the contiguous tensor can be collapsed to one
    // dimension, and the loop to translate the linear index to the array
    // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, DST, SRC)                                     \
    THCudaTensor_kernel_copy<TYPE, DST, SRC>                            \
      <<<grid, block>>>(dstInfo, srcInfo, (TYPE) totalElements);        \

#define HANDLE_SRC_CASE(TYPE, DST, SRC)         \
    {                                           \
      switch (SRC) {                            \
        case 1:                                 \
          HANDLE_CASE(TYPE, DST, 1);            \
          break;                                \
        case 2:                                 \
          HANDLE_CASE(TYPE, DST, 2);            \
          break;                                \
        case 3:                                 \
          HANDLE_CASE(TYPE, DST, 3);            \
          break;                                \
        case 4:                                 \
          HANDLE_CASE(TYPE, DST, 4);            \
          break;                                \
        case 5:                                 \
          HANDLE_CASE(TYPE, DST, 5);            \
          break;                                \
        default:                                \
          HANDLE_CASE(TYPE, -1, -1);            \
          break;                                \
      }                                         \
    }

#define HANDLE_DST_CASE(TYPE, DST, SRC)         \
    case DST:                                   \
      HANDLE_SRC_CASE(TYPE, DST, SRC);          \
      break;

    // Can we use 32-bit integer math in the kernel (the linear ID for the copy
    // and the resulting non-linear offset is all computable using 32-bit math?)
    // We also use unsigned index math in the kernel, as signed div/mod has
    // additional overhead.
    if (canUse32BitCopyMath(src) && canUse32BitCopyMath(dst)) {
      TensorInfo<unsigned int> dstInfo =
        THCudaTensor_computeTensorInfo<unsigned int>(dst);
      TensorInfo<unsigned int> srcInfo =
        THCudaTensor_computeTensorInfo<unsigned int>(src);

      switch (dstInfo.dims) {
        HANDLE_DST_CASE(unsigned int, 1, srcInfo.dims);
        HANDLE_DST_CASE(unsigned int, 2, srcInfo.dims);
        HANDLE_DST_CASE(unsigned int, 3, srcInfo.dims);
        HANDLE_DST_CASE(unsigned int, 4, srcInfo.dims);
        HANDLE_DST_CASE(unsigned int, 5, srcInfo.dims);
        default:
          HANDLE_DST_CASE(unsigned int, -1, srcInfo.dims);
      }
    } else {
      TensorInfo<unsigned long> dstInfo =
        THCudaTensor_computeTensorInfo<unsigned long>(dst);
      TensorInfo<unsigned long> srcInfo =
        THCudaTensor_computeTensorInfo<unsigned long>(src);

      switch (dstInfo.dims) {
        HANDLE_DST_CASE(unsigned long, 1, srcInfo.dims);
        HANDLE_DST_CASE(unsigned long, 2, srcInfo.dims);
        HANDLE_DST_CASE(unsigned long, 3, srcInfo.dims);
        HANDLE_DST_CASE(unsigned long, 4, srcInfo.dims);
        HANDLE_DST_CASE(unsigned long, 5, srcInfo.dims);
        default:
          HANDLE_DST_CASE(unsigned long, -1, srcInfo.dims);
      }
    }
#undef HANDLE_CASE
#undef HANDLE_SRC_CASE
#undef HANDLE_DST_CASE
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

#undef DIVUP
#undef LDG
#undef ELEMENTS_PER_THREAD
#undef THREADS_PER_BLOCK
#undef MAX_GRID_SIZE
#undef MAX_DIMS
