#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCBlas.h"
#include "THCTensorCopy.h"
#include "THCTensorRandom.h"
#include "THCApply.cuh"
#include "THCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

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

void THCudaTensor_indexCopy(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src)
{
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THCudaTensor_newContiguous(state, src);
  indices_ = THCudaTensor_newWithSize1d(state, nIndex);
  THCudaTensor_copyLong(state, indices_, indices);

  nRes = THCudaTensor_nElement(state, res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexCopy<<<nblocks, nthreads>>>(
    THCudaTensor_data(state, res_), THCudaTensor_data(state, src),
    stride_, THCudaTensor_data(state, indices_),
    res_->nDimension, dim, nIndex,
    THCudaTensor_nElement(state, src), res_->size[dim]
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(state, indices_);
  THCudaTensor_free(state, src);
}


void THCudaTensor_indexFill(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, float val)
{
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < res_->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(res_->nDimension > 0, 2, "Source tensor is empty");

  indices_ = THCudaTensor_newWithSize1d(state, nIndex);
  THCudaTensor_copyLong(state, indices_, indices);

  nRes = THCudaTensor_nElement(state, res_) / res_->size[dim] * nIndex;


  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, res_->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, res_->stride, res_->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexFill<<<nblocks, nthreads>>>(
    THCudaTensor_data(state, res_), stride_, THCudaTensor_data(state, indices_),
    res_->nDimension, dim, nIndex, nRes, res_->size[dim], val
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(state, indices_);
}

__global__ void THCudaTensor_kernel_indexSelect(
   float *tensor, float *src, long* src_stride, float *index,
   long src_nDim, int dim, long idx_size, long tensor_size, long size_dim
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
      int targetIdx = 0;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = src_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          srcIdx += coeff * src_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / src_stride[d];
          leftover -= coeff * src_stride[d];
          targetIdx += coeff * src_stride[d];
          srcIdx += coeff * src_stride[d];
        }
      }
      tensor[targetIdx + i*src_stride[dim]] = src[srcIdx + ((int)(index[i])-1)*src_stride[dim]];
    }
  }
}


void THCudaTensor_indexSelect(THCState *state, THCudaTensor *res_, THCudaTensor *src, int dim, THLongTensor *indices)
{
  THLongStorage *newSize;
  THCudaTensor *indices_;
  long *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize, src->size);
  newSize->data[dim] = nIndex;
  THCudaTensor_resize(state, res_, newSize, NULL);
  THLongStorage_free(newSize);

  indices_ = THCudaTensor_newWithSize1d(state, nIndex);
  THCudaTensor_copyLong(state, indices_, indices);

  nRes = THCudaTensor_nElement(state, res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  THCudaCheck(cudaMalloc((void**)&stride_, src->nDimension * sizeof(long)));
  THCudaCheck(cudaMemcpy(stride_, src->stride, src->nDimension * sizeof(long), cudaMemcpyHostToDevice));

  THCudaTensor_kernel_indexSelect<<<nblocks, nthreads>>>(
    THCudaTensor_data(state, res_), THCudaTensor_data(state, src),
    stride_, THCudaTensor_data(state, indices_),
    src->nDimension, dim, indices->size[0], nRes, src->size[dim]
  );

  THCudaCheck(cudaFree(stride_));
  THCudaTensor_free(state, indices_);
}
