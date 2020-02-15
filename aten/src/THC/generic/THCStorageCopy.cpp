#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorageCopy.cpp"
#else

void THCStorage_(copyCPU)(THCState *state, THCStorage *self, struct THStorage *src)
{
  THArgCheck(self->numel() == src->numel(), 2, "size does not match");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  THCudaCheck(cudaMemcpyAsync(THCStorage_(data)(state, self),
                              THStorage_(data)(src),
                              self->numel() * sizeof(scalar_t),
                              cudaMemcpyHostToDevice,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                          \
void THCStorage_(copy##TYPEC)(THCState *state, THCStorage *self, struct TH##TYPEC##Storage *src)  \
{                                                                      \
  THCTensor* selfTensor =                                              \
      THCTensor_(newWithStorage1d)(state, self, 0, self->numel(), 1);     \
  struct TH##TYPEC##Tensor* srcTensor =                                \
      TH##TYPEC##Tensor_newWithStorage1d(src, 0, src->numel(), 1);        \
  THCTensor_(copy)(state, selfTensor, srcTensor);               \
  TH##TYPEC##Tensor_free(srcTensor);                                   \
  THCTensor_(free)(state, selfTensor);                                 \
}
TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Float)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Half)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Bool)
TH_CUDA_STORAGE_IMPLEMENT_COPY(BFloat16)

void THStorage_(copyCuda)(THCState *state, THStorage *self, struct THCStorage *src)
{
  THArgCheck(self->numel() == src->numel(), 2, "size does not match");
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  THCudaCheck(cudaMemcpyAsync(THStorage_(data)(self),
                              THCStorage_(data)(state, src),
                              self->numel() * sizeof(scalar_t),
                              cudaMemcpyDeviceToHost,
                              stream));
  THCudaCheck(cudaStreamSynchronize(stream));
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                             \
void TH_CONCAT_4(TH,TYPEC,Storage_copyCuda,Real)(THCState *state, TH##TYPEC##Storage *self, struct THCStorage *src) \
{                                                                           \
  TH##TYPEC##Tensor* selfTensor =                                           \
      TH##TYPEC##Tensor_newWithStorage1d(self, 0, self->numel(), 1);           \
  struct THCTensor* srcTensor =                                             \
      THCTensor_(newWithStorage1d)(state, src, 0, src->numel(), 1);            \
  THCTensor_(copy)(state, selfTensor, srcTensor);                           \
  THCTensor_(free)(state, srcTensor);                                       \
  TH##TYPEC##Tensor_free(selfTensor);                                   \
}
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Float)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Half)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Bool)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(BFloat16)

#undef TH_CUDA_STORAGE_IMPLEMENT_COPY
#undef TH_CUDA_STORAGE_IMPLEMENT_COPYTO

#endif
