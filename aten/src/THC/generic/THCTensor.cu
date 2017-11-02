#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.cu"
#else

cudaTextureObject_t THCTensor_(getTextureObject)(THCState *state, THCTensor *self)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self));
  cudaTextureObject_t texObj;
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = THCTensor_(data)(state, self);
  resDesc.res.linear.sizeInBytes = THCTensor_(nElement)(state, self) * 4;
  resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
                                                  cudaChannelFormatKindFloat);
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    if (THCTensor_(nElement)(state, self) > 2>>27)
      THError("Failed to create texture object, "
              "nElement:%ld exceeds 27-bit addressing required for tex1Dfetch. Cuda Error: %s",
              THCTensor_(nElement)(state, self), cudaGetErrorString(errcode));
    else
      THError("Failed to create texture object: %s", cudaGetErrorString(errcode));
  }
  return texObj;
}

THC_API int THCTensor_(getDevice)(THCState* state, const THCTensor* tensor) {
  if (!tensor->storage) return -1;
  return THCStorage_(getDevice)(state, tensor->storage);
}

#endif
