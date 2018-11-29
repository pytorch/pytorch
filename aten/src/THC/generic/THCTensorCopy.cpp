#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorCopy.cpp"
#else

void THCTensor_(copyAsyncCPU)(THCState *state, THCTensor *self, struct THTensor *src)
{
  THArgCheck(THCTensor_(nElement)(state, self) == THTensor_(nElement)(src), 2, "sizes do not match");
  THArgCheck(THCTensor_(isContiguous)(state, self), 2, "Target tensor must be contiguous");
  THArgCheck(THTensor_(isContiguous)(src), 3, "Source tensor must be contiguous");

  if (THCTensor_(nElement)(state, self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCTensor_(getDevice)(state, self);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  THCudaCheck(cudaMemcpyAsync(THCTensor_(data)(state, self),
                              src->data<scalar_t>(),
                              THTensor_(nElement)(src) * sizeof(scalar_t),
                              cudaMemcpyHostToDevice,
                              stream.stream()));

  THCudaCheck(THCCachingHostAllocator_recordEvent(THStorage_(data)(THTensor_getStoragePtr(src)), stream));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

void THTensor_(copyAsyncCuda)(THCState *state, THTensor *self, struct THCTensor *src)
{
  THArgCheck(THTensor_(nElement)(self) == THCTensor_(nElement)(state, src), 2, "sizes do not match");
  THArgCheck(THTensor_(isContiguous)(self), 2, "Target tensor must be contiguous");
  THArgCheck(THCTensor_(isContiguous)(state, src), 3, "Source tensor must be contiguous");

  if (THTensor_(nElement)(self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THCTensor_(getDevice)(state, src);
  int currentDevice;
  THCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(tensorDevice));
  }

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  THCudaCheck(cudaMemcpyAsync(self->data<scalar_t>(),
                              THCTensor_(data)(state, src),
                              THCTensor_(nElement)(state, src) * sizeof(scalar_t),
                              cudaMemcpyDeviceToHost,
                              stream.stream()));

  THCudaCheck(THCCachingHostAllocator_recordEvent(THCStorage_(data)(state, THTensor_getStoragePtr(src)), stream));

  if (currentDevice != tensorDevice) {
    THCudaCheck(cudaSetDevice(currentDevice));
  }
}

#undef IMPLEMENT_TH_CUDA_TENSOR_COPY
#undef IMPLEMENT_TH_CUDA_TENSOR_COPY_TO

#endif
