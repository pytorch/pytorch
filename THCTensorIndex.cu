#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "THCDeviceUtils.cuh"
#include <algorithm> // for std::min

__global__ void THCudaTensor_kernel_indexFill(
   float *tensor, long* stride, float *index, long src_nDim,
   int dim, long idx_size, long tensor_size, long size_dim, float val
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = tensor_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          coeff = leftover / (stride[d] / size_dim);
          leftover -= coeff * (stride[d] / size_dim);
          srcIdx += coeff * stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / stride[d];
          leftover -= coeff * stride[d];
          srcIdx += coeff * stride[d];
        }
      }
        tensor[srcIdx + (long)((index[i])-1)*stride[dim]] = val;
    }
  }
}

__global__ void THCudaTensor_kernel_indexCopy(
   float *res, float *src, long* res_stride, float *index,
   long res_nDim, int dim, long idx_size, long src_size, long size_dim
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = src_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int resIdx = 0;
      for (int d=0; d<res_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = res_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          resIdx += coeff * res_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / res_stride[d];
          leftover -= coeff * res_stride[d];
          targetIdx += coeff * res_stride[d];
          resIdx += coeff * res_stride[d];
        }
      }
      res[resIdx + ((long)(index[i])-1)*res_stride[dim]] = src[targetIdx + i*res_stride[dim]];
    }
  }
}

__global__ void THCudaTensor_kernel_indexAdd(
   float *res, float *src, long* res_stride, float *index,
   long res_nDim, int dim, long idx_size, long src_size, long size_dim
)
{
  int thread_idx = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  long flat_size = src_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int resIdx = 0;
      for (int d=0; d<res_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = res_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          resIdx += coeff * res_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / res_stride[d];
          leftover -= coeff * res_stride[d];
          targetIdx += coeff * res_stride[d];
          resIdx += coeff * res_stride[d];
        }
      }
      atomicAdd(&res[resIdx + ((long)(index[i])-1)*res_stride[dim]], src[targetIdx + i*res_stride[dim]]);
    }
  }
}

void THCudaTensor_indexCopy_long(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 1, res_));

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexCopy(state, res_, dim, indices_, src);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexCopy(THCState *state, THCudaTensor *res_, int dim, THCudaTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, res_, src));
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THCudaTensor_newContiguous(state, src);
  indices = THCudaTensor_newContiguous(state, indices);

  nRes = THCudaTensor_nElement(state, res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(THCudaMalloc(state, (void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexCopy<<<nblocks, nthreads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, res_), THCudaTensor_data(state, src),
    stride_, THCudaTensor_data(state, indices),
    res_->nDimension, dim, nIndex,
    THCudaTensor_nElement(state, src), res_->size[dim]
  );

  THCudaCheck(THCudaFree(state, stride_));
  THCudaTensor_free(state, indices);
  THCudaTensor_free(state, src);
}

void THCudaTensor_indexAdd_long(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 1, res_));

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexAdd(state, res_, dim, indices_, src);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexAdd(THCState *state, THCudaTensor *res_, int dim, THCudaTensor *indices, THCudaTensor *src)
{
  THAssert(THCudaTensor_checkGPU(state, 2, res_, src));
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THCudaTensor_newContiguous(state, src);
  indices = THCudaTensor_newContiguous(state, indices);

  nRes = THCudaTensor_nElement(state, res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(THCudaMalloc(state, (void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexAdd<<<nblocks, nthreads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, res_), THCudaTensor_data(state, src),
    stride_, THCudaTensor_data(state, indices),
    res_->nDimension, dim, nIndex,
    THCudaTensor_nElement(state, src), res_->size[dim]
  );

  THCudaCheck(THCudaFree(state, stride_));
  THCudaTensor_free(state, indices);
  THCudaTensor_free(state, src);
}

void THCudaTensor_indexFill_long(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, float val)
{
  THAssert(THCudaTensor_checkGPU(state, 1, res_));

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexFill(state, res_, dim, indices_, val);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexFill(THCState *state, THCudaTensor *res_, int dim, THCudaTensor *indices, float val)
{
  THAssert(THCudaTensor_checkGPU(state, 1, res_));
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < res_->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(res_->nDimension > 0, 2, "Source tensor is empty");

  nRes = THCudaTensor_nElement(state, res_) / res_->size[dim] * nIndex;
  indices = THCudaTensor_newContiguous(state, indices);

  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(THCudaMalloc(state, (void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexFill<<<nblocks, nthreads, 0, THCState_getCurrentStream(state)>>>(
    THCudaTensor_data(state, res_), stride_, THCudaTensor_data(state, indices),
    res_->nDimension, dim, nIndex, nRes, res_->size[dim], val
  );

  THCudaCheck(THCudaFree(state, stride_));
  THCudaTensor_free(state, indices);
}

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectSmallIndex(TensorInfo<IndexType> dst,
                                      TensorInfo<IndexType> src,
                                      TensorInfo<IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {
    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<IndexType, IdxDim>::get(dstIndex, indices)] - 1;

    // We stride over the output ignoring the indexed dimension
    // (innerSize), whose offset calculation is handled differently
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
        IndexToOffset<IndexType, DstDim>::get(linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        IndexToOffset<IndexType, SrcDim>::get(linearIndex, src);
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
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__global__ void indexSelectLargeIndex(TensorInfo<IndexType> dst,
                                      TensorInfo<IndexType> src,
                                      TensorInfo<IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    // Lua indices begin at 1
    IndexType srcIndex =
      indices.data[IndexToOffset<IndexType, IdxDim>::get(dstIndex, indices)] - 1;

    IndexType dstOffset =
      IndexToOffset<IndexType, DstDim>::get(elementInSlice, dst);
    dstOffset += dstIndex * dst.strides[dstSelectDim];

    IndexType srcOffset =
      IndexToOffset<IndexType, SrcDim>::get(elementInSlice, src);
    srcOffset += srcIndex * src.strides[srcSelectDim];

    dst.data[dstOffset] = src.data[srcOffset];
  }
}

void THCudaTensor_indexSelect_long(THCState *state, THCudaTensor *res_, THCudaTensor *src, int dim, THLongTensor *indices)
{
  THAssert(THCudaTensor_checkGPU(state, 2, res_, src));

  THCudaTensor *indices_ = THCudaTensor_newWithSize1d(state, indices->size[0]);
  THCudaTensor_copyLong(state, indices_, indices);

  THCudaTensor_indexSelect(state, res_, src, dim, indices_);

  THCudaTensor_free(state, indices_);
}

void THCudaTensor_indexSelect(THCState *state, THCudaTensor *dst, THCudaTensor *src, int dim, THCudaTensor *indices)
{
  THAssert(THCudaTensor_checkGPU(state, 2, dst, src));

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
  long totalSize = THCudaTensor_nElement(state, dst);
  long sliceSize = totalSize / numIndices;

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;

#define SMALL_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)            \
  indexSelectSmallIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>        \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstSelectDim, srcSelectDim, totalSize, sliceSize);

#define LARGE_INDEX(TYPE, DST_DIM, SRC_DIM, IDX_DIM)            \
  indexSelectLargeIndex<TYPE, DST_DIM, SRC_DIM, IDX_DIM>        \
    <<<largeIndexGrid, largeIndexBlock, 0, stream>>>(           \
      dstInfo, srcInfo, indicesInfo,                            \
      dstSelectDim, srcSelectDim, totalSize, sliceSize);

  dim3 smallIndexGrid(std::min(THCCeilDiv(sliceSize, 128L), (long)(mpc * 8)));
  dim3 smallIndexBlock(std::min(sliceSize, 128L));

  dim3 largeIndexGrid(std::min(THCCeilDiv(totalSize, 128L), (long)(mpc * 8)));
  dim3 largeIndexBlock(std::min(totalSize, 128L));

  if (THC_canUse32BitIndexMath(state, dst) &&
      THC_canUse32BitIndexMath(state, src) &&
      THC_canUse32BitIndexMath(state, indices)) {
    TensorInfo<unsigned int> dstInfo(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.sizes[dstSelectDim] = 1;

    TensorInfo<unsigned int> srcInfo(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.sizes[srcSelectDim] = 1;

    TensorInfo<unsigned int> indicesInfo(state, indices);
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
    TensorInfo<unsigned long> dstInfo(state, dst);
    int dstSelectDim = dstInfo.collapseDims(dim);
    dstInfo.sizes[dstSelectDim] = 1;

    TensorInfo<unsigned long> srcInfo(state, src);
    int srcSelectDim = srcInfo.collapseDims(dim);
    srcInfo.sizes[srcSelectDim] = 1;

    TensorInfo<unsigned long> indicesInfo(state, indices);
    indicesInfo.collapseDims();

    LARGE_INDEX(unsigned long, -1, -1, -1);
  }

#undef SMALL_INDEX
#undef LARGE_INDEX
}
