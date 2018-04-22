#include "THC.h"
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCHalf.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCDeviceUtils.cuh"
#include "THCNumerics.cuh"
#include "THCAtomics.cuh"
#include "THCThrustAllocator.cuh"
#include "THCTensorSort.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm> // for std::min

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexCopyLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopySmallIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<T, IndexType> src,
                                    TensorInfo<int64_t, IndexType> indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType innerSize,
                                    int64_t dstCopyDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
    assert(dstIndex < dstCopyDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);

      dstOffset += dstIndex * dst.strides[dstCopyDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcCopyDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexCopySmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexCopyLargeIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<T, IndexType> src,
                                    TensorInfo<int64_t, IndexType> indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int64_t dstCopyDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
    assert(dstIndex < dstCopyDimSize);

    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstCopyDim];

    IndexType srcOffset =
      IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcCopyDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexAddLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddSmallIndex(TensorInfo<T, IndexType> dst,
                                   TensorInfo<T, IndexType> src,
                                   TensorInfo<int64_t, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   int64_t dstAddDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
    assert(dstIndex < dstAddDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexAddSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexAddLargeIndex(TensorInfo<T, IndexType> dst,
                                   TensorInfo<T, IndexType> src,
                                   TensorInfo<int64_t, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType totalSize,
                                   IndexType innerSize,
                                   int64_t dstAddDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex, elementInSlice;
    if (IndexIsMajor) {
      srcIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      srcIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;
    assert(dstIndex < dstAddDimSize);

    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstAddDim];

    IndexType srcOffset =
      IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcAddDim];

    atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFillLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillSmallIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<int64_t, IndexType> indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    int64_t dstFillDimSize,
                                    T val) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
    assert(dstIndex_ < dstFillDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex_ * dst.strides[dstFillDim];

      dst.data[dstOffset] = val;
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFillSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexFillLargeIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<int64_t, IndexType> indices,
                                    int dstFillDim,
                                    IndexType totalSize,
                                    IndexType innerSize,
                                    int64_t dstFillDimSize,
                                    T val) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex, elementInSlice;
    if (IndexIsMajor) {
      dstIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      dstIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
    assert(dstIndex_ < dstFillDimSize);

    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex_ * dst.strides[dstFillDim];

    dst.data[dstOffset] = val;
  }
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(TensorInfo<T, IndexType> dst,
                                      TensorInfo<T, IndexType> src,
                                      TensorInfo<int64_t, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
    assert(srcIndex < srcSelectDimSize);

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(linearIndex, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim,
          bool IndexIsMajor>
__global__ void indexSelectLargeIndex(TensorInfo<T, IndexType> dst,
                                      TensorInfo<T, IndexType> src,
                                      TensorInfo<int64_t, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      int64_t srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex, elementInSlice;
    if (IndexIsMajor) {
      dstIndex = linearIndex / innerSize;
      elementInSlice = linearIndex % innerSize;
    }
    else {
      elementInSlice = linearIndex / innerSize;
      dstIndex = linearIndex % innerSize;
    }

    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<int64_t, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;
    assert(srcIndex < srcSelectDimSize);

    IndexType dstOffset =
      IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

template <int Dims, typename T, typename IndexType>
__device__ __forceinline__ IndexType indexToOffset(
    const TensorInfo<T, IndexType>& info,
    int64_t index,
    IndexType size)
{
  IndexType linearIndex = static_cast<IndexType>(index);
  assert(linearIndex < size && linearIndex >= -size);
  if (linearIndex < 0) {
    linearIndex += size;
  }
  return IndexToOffset<T, IndexType, Dims>::get(linearIndex, info) - TH_INDEX_BASE;
}

struct WrapIndexOp {
  WrapIndexOp(int64_t size) : size(size) {}

  __device__ __forceinline__ void operator()(int64_t* out, int64_t* in) {
    auto idx = *in;
    assert(idx < size && idx >= -size);
    *out = idx < 0 ? idx + size : idx;
  }

  int64_t size;
};

template <typename T, typename IndexType, int Dims>
struct TensorTakeOp {
  TensorTakeOp(TensorInfo<T, IndexType> info, IndexType numel, int64_t*, int64_t*)
    : info(info), numel(numel) {}

  __device__ __forceinline__ void operator()(T* out, int64_t* index) {
    auto offset = indexToOffset<Dims>(info, *index, numel);
    *out = info.data[offset];
  }

  const TensorInfo<T, IndexType> info;
  IndexType numel;
};

template <typename T, typename IndexType, int Dims>
struct TensorPutOp {
  TensorPutOp(TensorInfo<T, IndexType> info, IndexType numel, int64_t*, int64_t*)
    : info(info), numel(numel) {}

  __device__ __forceinline__ void operator()(T* value, int64_t* index) {
    auto offset = indexToOffset<Dims>(info, *index, numel);
    info.data[offset] = *value;
  }

  const TensorInfo<T, IndexType> info;
  IndexType numel;
};

template <typename T, typename IndexType, int Dims>
struct TensorPutAccumulateOp {
  TensorPutAccumulateOp(TensorInfo<T, IndexType> info, IndexType numel, int64_t* start, int64_t* end)
    : info(info), numel(numel), start(start), end(end) {}

  __device__ __forceinline__ void operator()(T* value, int64_t* index) {
    if (index == start || *index != *(index - 1)) {
      int64_t linear_index = *index;
      auto offset = indexToOffset<Dims>(info, linear_index, numel);
      do {
        info.data[offset] = THCNumerics<T>::add(info.data[offset], *value);
        index++;
        value++;
      } while (index != end && *index == linear_index);
    }
  }

  const TensorInfo<T, IndexType> info;
  IndexType numel;
  int64_t* start;
  int64_t* end;
};


template<typename IndexType, typename real, template<class, class, int> class Op, typename TensorType>
void dispatchTakePutImpl(THCState *state, TensorType *a, TensorType *b, THCudaLongTensor *index) {
  // These are only valid if index is contiguous
  auto start = THCudaLongTensor_data(state, index);
  auto end = start + THCudaLongTensor_numel(state, index);

  auto aInfo = getTensorInfo<TensorType, IndexType>(state, a);
  aInfo.collapseDims();
  auto numel = TensorUtils<TensorType>::getNumElements(state, a);
  if (aInfo.isContiguous()) {
    auto op = Op<real, IndexType, -2>(aInfo, numel, start, end);
    THC_pointwiseApply2(state, b, index, op);
  } else {
    auto op = Op<real, IndexType, -1>(aInfo, numel, start, end);
    THC_pointwiseApply2(state, b, index, op);
  }
}

template<typename real, template<class, class, int> class Op, typename TensorType>
void dispatchTakePut(THCState *state, TensorType *a, TensorType *b, THCudaLongTensor *index) {
  if (TensorUtils<TensorType>::canUse32BitIndexMath(state, a, INT_MAX)) {
    dispatchTakePutImpl<int32_t, real, Op>(state, a, b, index);
  } else {
    dispatchTakePutImpl<int64_t, real, Op>(state, a, b, index);
  }
}

#include "generic/THCTensorIndex.cu"
#include "THCGenerateAllTypes.h"
