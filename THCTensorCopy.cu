#include "THGeneral.h"
#include "THCGeneral.h"
#include "THCTensor.h"
#include <assert.h>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

/* backward-compatible LDG */
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

template <typename IndexType>
TensorInfo<IndexType> THCudaTensor_computeTensorInfo(THCudaTensor* self) {
  int dims = THCudaTensor_nDimension(self);
  assert(dims <= MAX_DIMS);

  TensorInfo<IndexType> info;
  info.data = THCudaTensor_data(self);
  info.dims = dims;

  for (int i = 0; i < dims; ++i) {
    info.sizes[i] = (IndexType) THCudaTensor_size(self, i);
    info.strides[i] = (IndexType) THCudaTensor_stride(self, i);
  }

  for (int i = dims; i < MAX_DIMS; ++i) {
    info.sizes[i] = 1;
    info.strides[i] = info.strides[dims - 1] * info.sizes[dims - 1];
  }

  return info;
}

// Returns true if all linear ID -> offset math can be performed using 32 bit
// unsigned math
bool canUse32BitCopyMath(THCudaTensor* t) {
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
template <typename IndexType, int Dims>
__launch_bounds__(32 * 4, 16)
__global__ void THCudaTensor_kernel_copy(TensorInfo<IndexType> dst,
                                         TensorInfo<IndexType> src,
                                         IndexType totalElements) {
#define BDIM THREADS_PER_BLOCK

  const IndexType linearBlockId =
    blockIdx.z * gridDim.y * gridDim.x +
    blockIdx.y * gridDim.x +
    blockIdx.x;

  const IndexType startLinearId =
    linearBlockId * BDIM * ELEMENTS_PER_THREAD;

  IndexType endLinearId =
    (linearBlockId + 1) * BDIM * ELEMENTS_PER_THREAD;
  endLinearId = endLinearId < totalElements ? endLinearId : totalElements;

  for (IndexType linearId = startLinearId + threadIdx.x;
       linearId < endLinearId;
       linearId += BDIM) {
    // Convert `linearId` into an offset of `src`
    const IndexType srcOffset =
      linearIdToOffset<IndexType, Dims>(linearId, src);

    // Convert `linearId` into an offset of `dst`
    const IndexType dstOffset =
      linearIdToOffset<IndexType, Dims>(linearId, dst);

    dst.data[dstOffset] = LDG(&src.data[srcOffset]);
  }
}

THC_API void THCudaTensor_copy(THCudaTensor *self, THCudaTensor *src) {
  long totalElements = THCudaTensor_nElement(self);

  THArgCheck(totalElements == THCudaTensor_nElement(src), 2,
             "sizes do not match");

  THArgCheck(THCudaTensor_nDimension(self) <= MAX_DIMS, 2,
             "Copy only supported for <= 25 dimensions");
  THArgCheck(THCudaTensor_nDimension(src) <= MAX_DIMS, 3,
             "Copy only supported for <= 25 dimensions");

  if (THCudaTensor_nDimension(self) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }

  // We can memcpy the memory if both tensors are contiguous.
  // FIXME: also if both tensors have matching size and stride arrays with no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is contiguous).
  bool memcpyEligible =
    THCudaTensor_isContiguous(self) && THCudaTensor_isContiguous(src);

  if (memcpyEligible) {
    THCudaCheck(cudaMemcpyAsync(self->storage->data + self->storageOffset,
                                src->storage->data + src->storageOffset,
                                THCudaTensor_nElement(src) * sizeof(float),
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

    int largestDim =
      max(THCudaTensor_nDimension(self), THCudaTensor_nDimension(src));

#define HANDLE_CASE(N, TYPE)                                            \
      case N:                                                           \
        THCudaTensor_kernel_copy<TYPE, N>                               \
          <<<grid, block>>>(selfInfo, srcInfo, (TYPE) totalElements);   \
        break;

    // Can we use 32-bit integer math in the kernel (the linear ID for the copy
    // and the resulting non-linear offset is all computable using 32-bit math?)
    // We also use unsigned index math in the kernel, as signed div/mod has
    // additional overhead.
    if (canUse32BitCopyMath(src) && canUse32BitCopyMath(self)) {
      TensorInfo<unsigned int> selfInfo =
        THCudaTensor_computeTensorInfo<unsigned int>(self);
      TensorInfo<unsigned int> srcInfo =
        THCudaTensor_computeTensorInfo<unsigned int>(src);

      switch (largestDim) {
        HANDLE_CASE(1, unsigned int);
        HANDLE_CASE(2, unsigned int);
        HANDLE_CASE(3, unsigned int);
        HANDLE_CASE(4, unsigned int);
        HANDLE_CASE(5, unsigned int);
        HANDLE_CASE(6, unsigned int);
        HANDLE_CASE(7, unsigned int);
        HANDLE_CASE(8, unsigned int);
        default:
          THCudaTensor_kernel_copy<unsigned int, -1>
            <<<grid, block>>>(selfInfo, srcInfo, (unsigned int) totalElements);
          break;
      }
    } else {
      TensorInfo<unsigned long> selfInfo =
        THCudaTensor_computeTensorInfo<unsigned long>(self);
      TensorInfo<unsigned long> srcInfo =
        THCudaTensor_computeTensorInfo<unsigned long>(src);

      switch (largestDim) {
        HANDLE_CASE(1, unsigned long);
        HANDLE_CASE(2, unsigned long);
        HANDLE_CASE(3, unsigned long);
        HANDLE_CASE(4, unsigned long);
        HANDLE_CASE(5, unsigned long);
        HANDLE_CASE(6, unsigned long);
        HANDLE_CASE(7, unsigned long);
        HANDLE_CASE(8, unsigned long);
        default:
          THCudaTensor_kernel_copy<unsigned long, -1>
            <<<grid, block>>>(selfInfo, srcInfo, totalElements);
          break;
      }
    }

#undef HANDLE_CASE
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}
