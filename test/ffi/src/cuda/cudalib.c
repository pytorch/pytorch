#include <TH/TH.h>
#include <THC/THC.h>

extern THCState *state;

#include "../cpu/lib1.c"

void cuda_func(THCudaTensor *tensor, int a, float b)
{
  THCudaTensor_mul(state, tensor, tensor, a);
  THCudaTensor_add(state, tensor, tensor, b);
}
