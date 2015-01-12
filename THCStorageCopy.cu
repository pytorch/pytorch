#include "THCStorageCopy.h"
#include "THCGeneral.h"

void THCudaStorage_rawCopy(THCState *state, THCudaStorage *self, float *src)
{
  THCudaCheck(cudaMemcpyAsync(self->data, src, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void THCudaStorage_copy(THCState *state, THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void THCudaStorage_copyCuda(THCState *state, THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}
