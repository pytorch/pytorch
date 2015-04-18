#include "THCStorage.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

void THCudaStorage_fill(THCState *state, THCudaStorage *self, float value)
{
  thrust::device_ptr<float> self_data(self->data);
  thrust::fill(self_data, self_data+self->size, value);
}

void THCudaStorage_resize(THCState *state, THCudaStorage *self, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    return;

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCudaCheck(cudaFree(self->data));
    }
    self->data = NULL;
    self->size = 0;
  }
  else
  {
    int curDev;
    THCudaCheck(cudaGetDevice(&curDev));
    if(self->device != THC_DEVICE_NONE) {
      if (THCState_getDeviceMode(state) == THCStateDeviceModeAuto) {
        THCudaCheck(cudaSetDevice(self->device));
      }
      else if(self->device != curDev) {
        THError("THCudaStorage_resize: device mismatch: tensorDev=%d, curDev=%d", self->device + 1, curDev + 1);
      }
    }

    float *data = NULL;
    THCudaCheck(cudaMalloc((void**)(&data), size * sizeof(float)));

    if (self->data) {
      THCudaCheck(cudaMemcpyAsync(data,
                                  self->data,
                                  THMin(self->size, size) * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      THCudaCheck(cudaFree(self->data));
    }

    self->data = data;
    self->size = size;
    THCudaCheck(cudaGetDevice(&self->device));

    THCudaCheck(cudaSetDevice(curDev));
  }
}

int THCudaStorage_getDevice(THCState* state, const THCudaStorage *storage) {
  return storage->device;
}

void THCudaStorage_setDevice(THCState* state, THCudaStorage *storage, int device) {
  if(storage->size > 0 && storage->device != device) {
    THError("Cannot call setDevice() on a non-empty tensor. Use copy() instead.");
  }
  storage->device = device;
}