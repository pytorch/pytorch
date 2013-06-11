#include "THCTensor.h"
#include "THCGeneral.h"
#include "THGeneral.h"

static void THCudaTensor_computesz(THCudaTensor *self, long **sz_, long **st_)
{
  long *sz, *st, *szh;
  int i;
  
  THCudaCheck(cudaMalloc(&sz, sizeof(long)*self->nDimension));
  THCudaCheck(cudaMalloc(&st, sizeof(long)*self->nDimension));
  szh = (long*)THAlloc(sizeof(long)*self->nDimension);

  for(i = self->nDimension-1; i >= 0; i--)
  {
    if(i == self->nDimension-1)
      szh[i] = 1;
    else
      szh[i] = szh[i+1]*self->size[i+1];
  }

  THCudaCheck(cudaMemcpy(sz, szh, self->nDimension * sizeof(long), cudaMemcpyHostToDevice));
  THCudaCheck(cudaMemcpy(st, self->stride, self->nDimension * sizeof(long), cudaMemcpyHostToDevice));
  THFree(szh);

  *sz_ = sz;
  *st_ = st;
}

__global__ void THCudaTensor_kernel_copy(float *dst, 
                                         long *dst_sz, long *dst_st, int dst_dim,
                                         float *src,
                                         long *src_sz, long *src_st, int src_dim,
                                         long n_elem, long innerdim)
{
  long k = blockIdx.x*blockDim.y + threadIdx.y;

  long i_start = threadIdx.x * src_st[src_dim-1];
  long i_step = blockDim.x * src_st[src_dim-1];

  long o_start = threadIdx.x * dst_st[dst_dim-1];
  long o_step = blockDim.x * dst_st[dst_dim-1];
  long o_end = innerdim * dst_st[dst_dim-1];

  if ( ((k+1) * innerdim) <= n_elem) // too safe
  {
    long dst_idx = 0;
    long dst_rest = k * innerdim;
    for(int dim = 0; dim < dst_dim; dim++)
    {
      dst_idx += (dst_rest/dst_sz[dim])*dst_st[dim];
      dst_rest = dst_rest % dst_sz[dim];
    }

    long src_idx = 0;
    long src_rest = k * innerdim;
    for(int dim = 0; dim < src_dim; dim++)
    {
      src_idx += (src_rest/src_sz[dim])*src_st[dim];
      src_rest = src_rest % src_sz[dim];
    }

    for (int i=i_start, o=o_start; o<o_end; i+=i_step, o+=o_step) {
      dst[dst_idx + o] = src[src_idx + i];
    }
  }
}

void THCudaTensor_copy(THCudaTensor *self, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self) == THCudaTensor_nElement(src), 2, "sizes do not match"); 

  if(THCudaTensor_isContiguous(self) && THCudaTensor_isContiguous(src))
    THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src->storage->data + src->storageOffset, THCudaTensor_nElement(src) * sizeof(float), cudaMemcpyDeviceToDevice));
  else
  {    
    long *d_self_sz, *d_self_st, *d_src_sz, *d_src_st;
    long size = THCudaTensor_nElement(self);

    long ndims = self->nDimension;
    long innermostdim = self->size[ndims-1];

    THCudaTensor_computesz(self, &d_self_sz, &d_self_st);
    THCudaTensor_computesz(src, &d_src_sz, &d_src_st);

    int nblocks = ceil((float)size / (16 * innermostdim ));
    dim3 threads(16,16);
    dim3 grid(nblocks);

    THCudaTensor_kernel_copy<<<grid, threads>>>(THCudaTensor_data(self),
                                                d_self_sz, d_self_st, ndims,
                                                THCudaTensor_data(src),
                                                d_src_sz, d_src_st, src->nDimension,
                                                size, innermostdim);

    cudaError errcode = cudaGetLastError();
    if(errcode != cudaSuccess)
      THError(cudaGetErrorString(errcode));

    THCudaCheck(cudaFree(d_self_sz));
    THCudaCheck(cudaFree(d_self_st));
    THCudaCheck(cudaFree(d_src_sz));
    THCudaCheck(cudaFree(d_src_st));
  }
}
