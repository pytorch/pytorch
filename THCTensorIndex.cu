#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCDeviceUtils.cuh"
#include <algorithm> // for std::min

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexCopyLargeIndex kernel is a better choice to increase
// parallelism.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopySmallIndex(TensorInfo<float, IndexType> dst,
                                    TensorInfo<float, IndexType> src,
                                    TensorInfo<float, IndexType> indices,
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
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(srcIndex, indices)] - 1;

    if (dstIndex < dstCopyDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<float, IndexType, DstDim>::get(linearIndex, dst);

        dstOffset += dstIndex * dst.strides[dstCopyDim];

        IndexType srcOffset =
          IndexToOffset<float, IndexType, SrcDim>::get(linearIndex, src);
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
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexCopyLargeIndex(TensorInfo<float, IndexType> dst,
                                    TensorInfo<float, IndexType> src,
                                    TensorInfo<float, IndexType> indices,
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
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(srcIndex, indices)] - 1;

    if (dstIndex < dstCopyDimSize) {
      IndexType dstOffset =
        IndexToOffset<float, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstCopyDim];

      IndexType srcOffset =
        IndexToOffset<float, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcCopyDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

void THCudaTensor_indexCopy_long(THCState *state, THCudaTensor *dst, int dim, THLongTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, dst, src));

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexCopy(state, dst, dim, indices_, src);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexCopy(THCState *state, THCudaTensor *dst, int dim, THCudaTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 3, dst, indices, src));

  THCCheckTensorDims(state, dst, 2);
  THCCheckTensorDims(state, src, 5);
  THCCheckTensorDims(state, indices, 4);

  long numIndices = THCudaTensor_nElement(state, indices);

  long srcDims = THCudaTensor_nDimension(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");
  THArgCheck(numIndices == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  int indContig = THCudaTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  long srcTotalSize = THCudaTensor_nElement(state, src);
  long dstCopyDimSize = THCudaTensor_size(state, dst, dim);
  long sliceSize = srcTotalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)            \
  indexCopySmallIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>          \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstCopyDim, srcCopyDim, sliceSize, dstCopyDimSize);

#define LARGE_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)            \
  indexCopyLargeIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>          \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstCopyDim, srcCopyDim, sliceSize, dstCopyDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, 128L), (long)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, 128L));

  dim3 largeIndexGrid(std::min(THCCeilDiv(srcTotalSize, 128L), (long)(mpc * 8)));
  dim3 largeIndexBlock(std::min(srcTotalSize, 128L));

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<float, unsigned int> dstInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<float, unsigned int> srcInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<float, unsigned int> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(unsigned int, -1, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        LARGE_INDEX(unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        LARGE_INDEX(unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(unsigned int, -1, -1, -1);
      }
    }
  } else {
    TensorInfo<float, unsigned long> dstInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, dst);
    int dstCopyDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstCopyDim);

    TensorInfo<float, unsigned long> srcInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, src);
    int srcCopyDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcCopyDim);

    TensorInfo<float, unsigned long> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexAddLargeIndex kernel is a better choice to increase
// parallelism.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddSmallIndex(TensorInfo<float, IndexType> dst,
                                   TensorInfo<float, IndexType> src,
                                   TensorInfo<float, IndexType> indices,
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
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(srcIndex, indices)] - 1;

    if (dstIndex < dstAddDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<float, IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex * dst.strides[dstAddDim];

        IndexType srcOffset =
          IndexToOffset<float, IndexType, SrcDim>::get(linearIndex, src);
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
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexAddLargeIndex(TensorInfo<float, IndexType> dst,
                                   TensorInfo<float, IndexType> src,
                                   TensorInfo<float, IndexType> indices,
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
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(srcIndex, indices)] - 1;

    if (dstIndex < dstAddDimSize) {
      IndexType dstOffset =
        IndexToOffset<float, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
        IndexToOffset<float, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      atomicAdd(&dst.data[dstOffset], src.data[srcOffset]);
    }
  }
}

void THCudaTensor_indexAdd_long(THCState *state, THCudaTensor *dst, int dim, THLongTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, dst, src));

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexAdd(state, dst, dim, indices_, src);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexAdd(THCState *state, THCudaTensor *dst, int dim, THCudaTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 3, dst, indices, src));

  THCCheckTensorDims(state, dst, 2);
  THCCheckTensorDims(state, src, 5);
  THCCheckTensorDims(state, indices, 4);

  long numIndices = THCudaTensor_nElement(state, indices);

  long srcDims = THCudaTensor_nDimension(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");
  THArgCheck(numIndices == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  int indContig = THCudaTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  long srcTotalSize = THCudaTensor_nElement(state, src);
  long dstAddDimSize = THCudaTensor_size(state, dst, dim);
  long sliceSize = srcTotalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)    \
  indexAddSmallIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>   \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(   \
      dstInfo, srcInfo, indicesInfo,                    \
      dstAddDim, srcAddDim, sliceSize, dstAddDimSize);

#define LARGE_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)    \
  indexAddLargeIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>   \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(   \
      dstInfo, srcInfo, indicesInfo,                    \
      dstAddDim, srcAddDim, sliceSize, dstAddDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, 128L), (long)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, 128L));

  dim3 largeIndexGrid(std::min(THCCeilDiv(srcTotalSize, 128L), (long)(mpc * 8)));
  dim3 largeIndexBlock(std::min(srcTotalSize, 128L));

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<float, unsigned int> dstInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, dst);
    int dstAddDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstAddDim);

    TensorInfo<float, unsigned int> srcInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, src);
    int srcAddDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcAddDim);

    TensorInfo<float, unsigned int> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(unsigned int, -1, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        LARGE_INDEX(unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        LARGE_INDEX(unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(unsigned int, -1, -1, -1);
      }
    }
  } else {
    TensorInfo<float, unsigned long> dstInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, dst);
    int dstAddDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstAddDim);

    TensorInfo<float, unsigned long> srcInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, src);
    int srcAddDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcAddDim);

    TensorInfo<float, unsigned long> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexFillLargeIndex kernel is a better choice to increase
// parallelism.
template <typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillSmallIndex(TensorInfo<float, IndexType> dst,
                                    TensorInfo<float, IndexType> indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    float val) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(dstIndex, indices)] - 1;

    if (dstIndex < dstFillDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<float, IndexType, DstDim>::get(linearIndex, dst);
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
template <typename IndexType, int DstDim, int IdxDim>
__global__ void indexFillLargeIndex(TensorInfo<float, IndexType> dst,
                                    TensorInfo<float, IndexType> indices,
                                    int dstFillDim,
                                    IndexType innerSize,
                                    long dstFillDimSize,
                                    float val) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < innerSize * indices.sizes[0];
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType dstIndex_ =
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(dstIndex, indices)] - 1;

    if (dstIndex_ < dstFillDimSize) {
      IndexType dstOffset =
        IndexToOffset<float, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex_ * dst.strides[dstFillDim];

      dst.data[dstOffset] = val;
    }
  }
}

void THCudaTensor_indexFill_long(THCState *state, THCudaTensor *dst, int dim, THLongTensor *indices, float val)
{
  THAssert(THCudaTensor_checkGPU(state, 1, dst));

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexFill(state, dst, dim, indices_, val);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexFill(THCState *state, THCudaTensor *dst, int dim, THCudaTensor *indices, float val)
{
  THAssert(THCudaTensor_checkGPU(state, 2, dst, indices));
  THCCheckTensorDims(state, dst, 2);
  THCCheckTensorDims(state, indices, 4);

  long numIndices = THCudaTensor_nElement(state, indices);

  long srcDims = THCudaTensor_nDimension(state, dst);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  int indContig = THCudaTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  long dstTotalSize = THCudaTensor_nElement(state, dst);
  long dstFillDimSize = THCudaTensor_size(state, dst, dim);
  long sliceSize = dstTotalSize / dstFillDimSize;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TYPE, DST_DIM, IDX_DIM)             \
  indexFillSmallIndex<TYPE, DST_DIM, IDX_DIM>           \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(   \
      dstInfo, indicesInfo,                             \
      dstFillDim, sliceSize, dstFillDimSize, val);

#define LARGE_INDEX(TYPE, DST_DIM, IDX_DIM)             \
  indexFillLargeIndex<TYPE, DST_DIM, IDX_DIM>           \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(   \
      dstInfo, indicesInfo,                             \
      dstFillDim, sliceSize, dstFillDimSize, val);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, 128L), (long)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, 128L));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, 128L), (long)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, 128L));

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<float, unsigned int> dstInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<float, unsigned int> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && indContig) {
        SMALL_INDEX(unsigned int, 1, -2);
      } else if (dstInfo.dims == 2 && indContig) {
        SMALL_INDEX(unsigned int, 2, -2);
      } else if (dstInfo.dims == 3 && indContig) {
        SMALL_INDEX(unsigned int, 3, -2);
      } else {
        SMALL_INDEX(unsigned int, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && indContig) {
        LARGE_INDEX(unsigned int, 1, -2);
      } else if (dstInfo.dims == 2 && indContig) {
        LARGE_INDEX(unsigned int, 2, -2);
      } else if (dstInfo.dims == 3 && indContig) {
        LARGE_INDEX(unsigned int, 3, -2);
      } else {
        LARGE_INDEX(unsigned int, -1, -1);
      }
    }
  } else {
    TensorInfo<float, unsigned long> dstInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, dst);
    int dstFillDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstFillDim);

    TensorInfo<float, unsigned long> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(unsigned long, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(TensorInfo<float, IndexType> dst,
                                      TensorInfo<float, IndexType> src,
                                      TensorInfo<float, IndexType> indices,
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
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(dstIndex, indices)] - 1;

    if (srcIndex < srcSelectDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<float, IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex * dst.strides[dstSelectDim];

        IndexType srcOffset =
          IndexToOffset<float, IndexType, SrcDim>::get(linearIndex, src);
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
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectLargeIndex(TensorInfo<float, IndexType> dst,
                                      TensorInfo<float, IndexType> src,
                                      TensorInfo<float, IndexType> indices,
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
      indices.data[IndexToOffset<float, IndexType, IdxDim>::get(dstIndex, indices)] - 1;

    if (srcIndex < srcSelectDimSize) {
      IndexType dstOffset =
        IndexToOffset<float, IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        IndexToOffset<float, IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

void THCudaTensor_indexSelect_long(THCState *state, THCudaTensor *dst, THCudaTensor *src, int dim, THLongTensor *indices)
{
  THAssert(THCudaTensor_checkGPU(state, 2, dst, src));

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexSelect(state, dst, src, dim, indices_);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexSelect(THCState *state, THCudaTensor *dst, THCudaTensor *src, int dim, THCudaTensor *indices)
{
  THAssert(THCudaTensor_checkGPU(state, 3, dst, src, indices));

  THCCheckTensorDims(state, dst, 2);
  THCCheckTensorDims(state, src, 3);
  THCCheckTensorDims(state, indices, 5);

  long numIndices = THCudaTensor_nElement(state, indices);

  long srcDims = THCudaTensor_nDimension(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  THLongStorage *newSize = THCudaTensor_newSizeOf(state, src);
  THLongStorage_set(newSize, dim, numIndices);
  THCudaTensor_resize(state, dst, newSize, NULL);
  THLongStorage_free(newSize);

  int indContig = THCudaTensor_isContiguous(state, indices);

  // The `src` is partitioned into two parts:
  // -the size of each slice we are indexing, which is the
  // total size of the tensor ignoring dimension `dim`;
  // -the number of indices we are choosing, which is the total size
  // of the tensor `indices`.
  long dstTotalSize = THCudaTensor_nElement(state, dst);
  long srcSelectDimSize = THCudaTensor_size(state, src, dim);
  long sliceSize = dstTotalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)            \
  indexSelectSmallIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>        \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstSelectDim, srcSelectDim, sliceSize, srcSelectDimSize);

#define LARGE_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)                    \
  indexSelectLargeIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>                \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(                   \
      dstInfo, srcInfo, indicesInfo,                                    \
      dstSelectDim, srcSelectDim, dstTotalSize, sliceSize, srcSelectDimSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, 128L), (long)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, 128L));

  dim3 largeIndexGrid(std::min(THCCeilDiv(dstTotalSize, 128L), (long)(mpc * 8)));
  dim3 largeIndexBlock(std::min(dstTotalSize, 128L));

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, dst) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, indices)) {
    TensorInfo<float, unsigned int> dstInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<float, unsigned int> srcInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<float, unsigned int> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, indices);
    indicesInfo.collapseDims();

    // A reasonable choice for when to have each thread iterate over
    // indices to choose
    if (numIndices <= 16) {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        SMALL_INDEX(unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        SMALL_INDEX(unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        SMALL_INDEX(unsigned int, 3, 3, -2);
      } else {
        SMALL_INDEX(unsigned int, -1, -1, -1);
      }
    } else {
      if (dstInfo.dims == 1 && srcInfo.dims == 1 && indContig) {
        LARGE_INDEX(unsigned int, 1, 1, -2);
      } else if (dstInfo.dims == 2 && srcInfo.dims == 2 && indContig) {
        LARGE_INDEX(unsigned int, 2, 2, -2);
      } else if (dstInfo.dims == 3 && srcInfo.dims == 3 && indContig) {
        LARGE_INDEX(unsigned int, 3, 3, -2);
      } else {
        LARGE_INDEX(unsigned int, -1, -1, -1);
      }
    }
  } else {
    TensorInfo<float, unsigned long> dstInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.reduceDim(dstSelectDim);

    TensorInfo<float, unsigned long> srcInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.reduceDim(srcSelectDim);

    TensorInfo<float, unsigned long> indicesInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}
