#include "THCTensor.h"

cudaTextureObject_t THCudaTensor_getTextureObject(THCudaTensor *self)
{
  cudaTextureObject_t texObj;
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = THCudaTensor_data(self);
  resDesc.res.linear.sizeInBytes = THCudaTensor_nElement(self) * 4;
  resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, 
                                                  cudaChannelFormatKindFloat);
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));
  return texObj;
}
