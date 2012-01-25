#include "THCStorage.h"

#include <thrust/fill.h>

void THCudaStorage_fill(THCudaStorage *self, float value)
{
  thrust::device_ptr<float> self_data(self->data);
  thrust::fill(self_data, self_data+self->size, value);
}
