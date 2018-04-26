#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensor.cu"
#else

THC_API int THCTensor_(getDevice)(THCState* state, const THCTensor* tensor) {
  if (!tensor->storage) return -1;
  return THCStorage_(getDevice)(state, tensor->storage);
}

#endif
