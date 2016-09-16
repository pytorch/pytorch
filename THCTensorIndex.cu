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
                                    TensorInfo<long, IndexType> indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType innerSize,
                                    long dstCopyDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstCopyDimSize) {
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
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexCopySmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopyLargeIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<T, IndexType> src,
                                    TensorInfo<long, IndexType> indices,
                                    int dstCopyDim,
                                    int srcCopyDim,
                                    IndexType innerSize,
                                    long dstCopyDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstCopyDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstCopyDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcCopyDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

template <typename T, size_t n>
struct AtomicAddIntegerImpl;

template<typename T>
struct AtomicAddIntegerImpl<T, 1> {
  __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui =
        (unsigned int *) (address - ((size_t)address & 3));
    unsigned int old = *address_as_ui;
    unsigned int shift = (((size_t)address & 3) * 8);
    unsigned int sum;
    unsigned int assumed;

    do {
      assumed = old;
      sum = val + T((old >> shift) & 0xff);
      old = (old & ~(0x000000ff << shift)) | (sum << shift);
      old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 2> {
  __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui =
        (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int sum;
    unsigned int newval;
    unsigned int assumed;

    do {
      assumed = old;
      sum = val + (size_t)address & 2 ? T(old >> 16) : T(old & 0xffff);
      newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16) : (old & 0xffff0000) | sum;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 4> {
  __device__ void operator()(T *address, T val) {
    unsigned int * address_as_ui = (unsigned int *) (address);
    unsigned int old = *address_as_ui;
    unsigned int newval;
    unsigned int assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

template<typename T>
struct AtomicAddIntegerImpl<T, 8> {
  __device__ void operator()(T *address, T val) {
    unsigned long long * address_as_ui = (unsigned long long *) (address);
    unsigned long long old = *address_as_ui;
    unsigned long long newval;
    unsigned long long assumed;

    do {
      assumed = old;
      newval = val +  (T)old;
      old = atomicCAS(address_as_ui, assumed, newval);
    } while (assumed != old);
  }
};

__device__ void atomicAdd(unsigned char *address, unsigned char val) {
  AtomicAddIntegerImpl<unsigned char, sizeof(unsigned char)>()(address, val);
}

__device__ void atomicAdd(char *address, char val) {
  AtomicAddIntegerImpl<char, sizeof(char)>()(address, val);
}

__device__ void atomicAdd(short *address, short val) {
  AtomicAddIntegerImpl<short, sizeof(short)>()(address, val);
}

__device__ void atomicAdd(long *address, long val) {
  AtomicAddIntegerImpl<long, sizeof(long)>()(address, val);
}

#ifdef CUDA_HALF_TENSOR
__device__ void atomicAdd(half *address, half val) {
  unsigned int * address_as_ui =
      (unsigned int *) ((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = THCNumerics<half>::add(hsum, val);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
   } while (assumed != old);
}
#endif

// from CUDA C Programmic Guide
__device__  void atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
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
                                   TensorInfo<long, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   long dstAddDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstAddDimSize) {
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
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexAddSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddLargeIndex(TensorInfo<T, IndexType> dst,
                                   TensorInfo<T, IndexType> src,
                                   TensorInfo<long, IndexType> indices,
                                   int dstAddDim,
                                   int srcAddDim,
                                   IndexType innerSize,
                                   long dstAddDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType srcIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(srcIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstAddDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
    }
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
                                    TensorInfo<long, IndexType> indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    T val) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex < dstFillDimSize) {
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
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexFillSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillLargeIndex(TensorInfo<T, IndexType> dst,
                                    TensorInfo<long, IndexType> indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    T val) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (dstIndex_ < dstFillDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex_ * dst.strides[dstFillDim];

      dst.data[dstOffset] = val;
    }
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
                                      TensorInfo<long, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      long srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (srcIndex < srcSelectDimSize) {
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
}

// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename T, typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectLargeIndex(TensorInfo<T, IndexType> dst,
                                      TensorInfo<T, IndexType> src,
                                      TensorInfo<long, IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      long srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<long, IndexType, IdxDim>::get(dstIndex, indices)] - TH_INDEX_BASE;

    if (srcIndex < srcSelectDimSize) {
      IndexType dstOffset =
        IndexToOffset<T, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        IndexToOffset<T, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

#include "generic/THCTensorIndex.cu"
#include "THCGenerateAllTypes.h"
