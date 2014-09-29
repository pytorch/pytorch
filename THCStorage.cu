#include "THCStorage.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

void THCudaStorage_fill(THCudaStorage *self, float value)
{
  thrust::device_ptr<float> self_data(self->data);
  thrust::fill(self_data, self_data+self->size, value);
}

void THCudaStorage_resize(THCudaStorage *self, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    return;

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM)
      THCudaCheck(cudaFree(self->data));
    self->data = NULL;
    self->size = 0;
  }
  else
  {
    float *data;
    THCudaCheck(cudaMalloc((void**)(&data), size * sizeof(float)));
    THCudaCheck(cudaMemcpyAsync(data, self->data, THMin(self->size, size) * sizeof(float), cudaMemcpyDeviceToDevice));
    THCudaCheck(cudaFree(self->data));    
    self->data = data;
    self->size = size;
  }  
}

void THCudaStorage_rawCopy(THCudaStorage *self, float *src)
{
  THCudaCheck(cudaMemcpyAsync(self->data, src, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void THCudaStorage_copy(THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void THCudaStorage_copyCuda(THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}
