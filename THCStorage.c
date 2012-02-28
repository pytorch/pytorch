#include "THCStorage.h"
#include "THCGeneral.h"

void THCudaStorage_set(THCudaStorage *self, long index, float value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  THCudaCheck(cudaMemcpy(self->data + index, &value, sizeof(float), cudaMemcpyHostToDevice));
}

float THCudaStorage_get(const THCudaStorage *self, long index)
{
  float value;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  THCudaCheck(cudaMemcpy(&value, self->data + index, sizeof(float), cudaMemcpyDeviceToHost));
  return value;
}

THCudaStorage* THCudaStorage_new(void)
{
  THCudaStorage *storage = THAlloc(sizeof(THCudaStorage));
  storage->data = NULL;
  storage->size = 0;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THCudaStorage* THCudaStorage_newWithSize(long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
  {
    THCudaStorage *storage = THAlloc(sizeof(THCudaStorage));
    THCudaCheck(cudaMalloc((void**)&(storage->data), size * sizeof(float)));
    storage->size = size;
    storage->refcount = 1;
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    return storage;
  }
  else
  {
    return THCudaStorage_new();
  }
}

THCudaStorage* THCudaStorage_newWithSize1(float data0)
{
  THCudaStorage *self = THCudaStorage_newWithSize(1);
  THCudaStorage_set(self, 0, data0);
  return self;
}

THCudaStorage* THCudaStorage_newWithSize2(float data0, float data1)
{
  THCudaStorage *self = THCudaStorage_newWithSize(2);
  THCudaStorage_set(self, 0, data0);
  THCudaStorage_set(self, 1, data1);
  return self;
}

THCudaStorage* THCudaStorage_newWithSize3(float data0, float data1, float data2)
{
  THCudaStorage *self = THCudaStorage_newWithSize(3);
  THCudaStorage_set(self, 0, data0);
  THCudaStorage_set(self, 1, data1);
  THCudaStorage_set(self, 2, data2);
  return self;
}

THCudaStorage* THCudaStorage_newWithSize4(float data0, float data1, float data2, float data3)
{
  THCudaStorage *self = THCudaStorage_newWithSize(4);
  THCudaStorage_set(self, 0, data0);
  THCudaStorage_set(self, 1, data1);
  THCudaStorage_set(self, 2, data2);
  THCudaStorage_set(self, 3, data3);
  return self;
}

THCudaStorage* THCudaStorage_newWithMapping(const char *fileName, int isShared)
{
  THError("not available yet for THCudaStorage");
  return NULL;
}

void THCudaStorage_retain(THCudaStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    ++self->refcount;
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
    THCudaCheck(cudaMemcpy(data, self->data, THMin(self->size, size) * sizeof(float), cudaMemcpyDeviceToDevice));
    THCudaCheck(cudaFree(self->data));    
    self->data = data;
    self->size = size;
  }  
}

void THCudaStorage_free(THCudaStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (--(self->refcount) <= 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM)
      THCudaCheck(cudaFree(self->data));
    THFree(self);
  }
}

void THCudaStorage_rawCopy(THCudaStorage *self, float *src)
{
  THCudaCheck(cudaMemcpy(self->data, src, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void THCudaStorage_copy(THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void THCudaStorage_copyCuda(THCudaStorage *self, THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToDevice));
}

void THCudaStorage_copyFloat(THCudaStorage *self, struct THFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(float), cudaMemcpyHostToDevice));
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                           \
  void THCudaStorage_copy##TYPEC(THCudaStorage *self, struct TH##TYPEC##Storage *src) \
  {                                                                     \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    THFloatStorage *buffer = THFloatStorage_newWithSize(src->size);     \
    THFloatStorage_copy##TYPEC(buffer, src);                            \
    THCudaStorage_copyFloat(self, buffer);                              \
    THFloatStorage_free(buffer);                                        \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)

void THFloatStorage_copyCuda(THFloatStorage *self, struct THCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToHost));
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                           \
  void TH##TYPEC##Storage_copyCuda(TH##TYPEC##Storage *self, struct THCudaStorage *src) \
  {                                                                     \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    THFloatStorage *buffer = THFloatStorage_newWithSize(src->size);     \
    THFloatStorage_copyCuda(buffer, src);                               \
    TH##TYPEC##Storage_copyFloat(self, buffer);                         \
    THFloatStorage_free(buffer);                                        \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)
