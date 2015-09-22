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
        tensor[srcIdx + (int)((index[i])-1)*stride[dim]] = val;
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
      res[resIdx + ((int)(index[i])-1)*res_stride[dim]] = src[targetIdx + i*res_stride[dim]];
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

__global__ void THCudaTensor_kernel_indexSelect_contiguous(
  float *tensor, float *src, long stride, float *index, long idxSize)
{
  // In the typical case, each block of 128 threads handles a 4x128
  // section of the output with each warp handling a single 1x128 row.
  // The outer loops handle inputs larger than 4*65535 or strides larger
  // than 128*65535.
  const int VT = 4;
  const int WARP_SIZE = 32;
  const int MAX_DIM_SIZE = 65535;

  for (int idx = blockIdx.x * blockDim.y + threadIdx.y; idx < idxSize; idx += blockDim.y * MAX_DIM_SIZE) {
    for (int startIdx = threadIdx.x + blockIdx.y * VT*WARP_SIZE; startIdx < stride; startIdx += VT*WARP_SIZE*MAX_DIM_SIZE) {
      const int srcIdx = ((int) index[idx] - 1) * stride;
      const int targetIdx = idx * stride;

      #pragma unroll
      for (int i = 0; i < VT; i++) {
        const int featureIdx = startIdx + i * WARP_SIZE;
        if (featureIdx < stride) {
          tensor[targetIdx + featureIdx] = src[srcIdx + featureIdx];
        }
      }
    }
  }
}

__global__ void THCudaTensor_kernel_indexSelect(
  TensorInfo<unsigned long> dest,
  TensorInfo<unsigned long> src,
  TensorInfo<unsigned long> indices,
  long totalSize,
  int indexDim
)
{
  for (unsigned long linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {

    unsigned long destOffset =
      IndexToOffset<unsigned long, -1>::get(linearIndex, dest);

    // Calculate the index in the dimension we're selecting in.
    unsigned long offset = destOffset;

    // In the process of doing so, we'll calculate indices in all lower
    // dimensions. We need to save these so we can reconstruct a linear index in
    // the source tensor. MAX_CUTORCH_DIMS is usually an overestimate (we only
    // need to save dims - indexDim indices) but this avoids dynamic memory
    // allocation.
    unsigned long unraveledIndices[MAX_CUTORCH_DIMS];

    for (int i = dest.dims - 1; i >= indexDim; --i) {
      unraveledIndices[i] = offset % dest.sizes[i];
      offset /= dest.sizes[i];
    }

    unsigned long destSliceIndex = unraveledIndices[indexDim];
    unsigned long destSliceOffset =
      IndexToOffset<unsigned long, 1>::get(destSliceIndex, indices);
    unsigned long srcSliceIndex =
      (unsigned long)indices.data[destSliceOffset] - 1;

    // Rebuild index in the source tensor by doing the reverse of the above
    unsigned long srcIndex = offset * src.sizes[indexDim] + srcSliceIndex;
    for (int i = indexDim + 1; i < dest.dims; ++i) {
      srcIndex = srcIndex * src.sizes[i] + unraveledIndices[i];
    }

    unsigned long srcOffset =
      IndexToOffset<unsigned long, -1>::get(srcIndex, src);
    dest.data[destOffset] = src.data[srcOffset];
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

void THCudaTensor_indexSelect(THCState *state, THCudaTensor *res, THCudaTensor *src, int dim, THCudaTensor *indices)
{
  THAssert(THCudaTensor_checkGPU(state, 2, res, src));

  THCCheckTensorDims(state, res, 2);
  THCCheckTensorDims(state, src, 3);
  THCCheckTensorDims(state, indices, 5);

  long nIndex = THCudaTensor_size(state, indices, 0);
  long srcDims = THCudaTensor_nDimension(state, src);
  cudaStream_t stream = THCState_getCurrentStream(state);

  THArgCheck(THCudaTensor_nDimension(state, indices) == 1, 3,
             "expecting vector of indices");
  THArgCheck(dim < srcDims, 4, "Indexing dim is out of bounds");
  THArgCheck(srcDims > 0, 2, "Source tensor is empty");

  THLongStorage *newSize = THCudaTensor_newSizeOf(state, src);
  THLongStorage_set(newSize, dim, nIndex);
  THCudaTensor_resize(state, res, newSize, NULL);
  THLongStorage_free(newSize);

  if (THCudaTensor_isContiguous(state, src) &&
      THCudaTensor_isContiguous(state, res) &&
      THCudaTensor_isContiguous(state, indices) &&
      dim == 0)
  {
    long stride = THCudaTensor_stride(state, src, 0);

    int blockX = std::min(THCCeilDiv(nIndex, 4L), 65535L);
    int blockY = std::min(THCCeilDiv(stride, 128L), 65535L);

    dim3 nthreads(32, 4);
    dim3 nblocks(blockX, blockY);

    THCudaTensor_kernel_indexSelect_contiguous<<<nblocks, nthreads, 0, stream>>>(
      THCudaTensor_data(state, res),
      THCudaTensor_data(state, src),
      stride,
      THCudaTensor_data(state, indices),
      nIndex);

    return;
  }

  long nRes = THCudaTensor_nElement(state, res);

  int mpc = THCState_getCurrentDeviceProperties(state)->multiProcessorCount;
  dim3 nthreads(std::min(nRes, 128L));
  dim3 nblocks(std::min(THCCeilDiv(nRes, 128L), (long)(mpc * 8)));

  TensorInfo<unsigned long> destInfo(state, res, NoCollapseDims);
  TensorInfo<unsigned long> srcInfo(state, src, NoCollapseDims);
  TensorInfo<unsigned long> indicesInfo(state, indices);

  THCudaTensor_kernel_indexSelect<<<nthreads, nblocks, 0, stream>>>(
    destInfo,
    srcInfo,
    indicesInfo,
    nRes,
    dim
  );
}
