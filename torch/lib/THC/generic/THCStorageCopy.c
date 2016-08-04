#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.c"
#else

#ifndef THC_REAL_IS_HALF
void THCStorage_(copyCPU)(THCState *state, THCStorage *self, struct THStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(real), cudaMemcpyHostToDevice));
}
#endif

#ifndef THC_REAL_IS_HALF
#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                            \
  void THCStorage_(copy##TYPEC)(THCState *state, THCStorage *self, struct TH##TYPEC##Storage *src)  \
  {                                                                      \
    if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {  \
      THCStorage_(copyCPU)(state, self, (THStorage*) src);   /* cast just removes compiler warning */ \
    } else {                                                             \
      THStorage *buffer;                                                 \
      THArgCheck(self->size == src->size, 2, "size does not match");     \
      buffer = THStorage_(newWithSize)(src->size);                       \
      THStorage_(copy##TYPEC)(buffer, src);                              \
      THCStorage_(copyCPU)(state, self, buffer);                         \
      THStorage_(free)(buffer);                                          \
    }                                                                    \
  }
#else
#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                            \
  void THCStorage_(copy##TYPEC)(THCState *state, THCStorage *self, struct TH##TYPEC##Storage *src)  \
  {                                                                      \
    THArgCheck(self->size == src->size, 2, "size does not match");       \
    THCudaStorage *buffer = THCudaStorage_newWithSize(state, src->size); \
    THCudaStorage_copy##TYPEC(state, buffer, src);                       \
    THCFloat2Half(state, self->data, buffer->data, src->size);           \
    THCudaStorage_free(state, buffer);                                   \
  }
#endif

TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Float)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)

#ifndef THC_REAL_IS_HALF
void THStorage_(copyCuda)(THCState *state, THStorage *self, struct THCStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(real), cudaMemcpyDeviceToHost));
}
#endif

#ifndef THC_REAL_IS_HALF
#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                         \
  void TH_CONCAT_4(TH,TYPEC,Storage_copyCuda,Real)(THCState *state, TH##TYPEC##Storage *self, struct THCStorage *src) \
  {                                                                     \
    if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) { \
      THStorage_(copyCuda)(state, (THStorage*) self, src); /* cast just removes compiler warnings */                    \
    } else {                                                            \
      THStorage *buffer;                                                \
      THArgCheck(self->size == src->size, 2, "size does not match");    \
      buffer = THStorage_(newWithSize)(src->size);                      \
      THStorage_(copyCuda)(state, buffer, src);                         \
      TH_CONCAT_4(TH,TYPEC,Storage_copy,Real)(self, buffer);            \
      THStorage_(free)(buffer);                                         \
    }                                                                   \
  }
#else
#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                         \
  void TH_CONCAT_4(TH,TYPEC,Storage_copyCuda,Real)(THCState *state, TH##TYPEC##Storage *self, struct THCStorage *src) \
  {                                                                     \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    THCudaStorage *buffer = THCudaStorage_newWithSize(state, src->size);\
    THCHalf2Float(state, buffer->data, src->data, src->size);           \
    TH_CONCAT_3(TH,TYPEC,Storage_copyCudaFloat)(state, self, buffer);   \
    THCudaStorage_free(state, buffer);                                  \
  }
#endif

TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Float)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)

#undef TH_CUDA_STORAGE_IMPLEMENT_COPY
#undef TH_CUDA_STORAGE_IMPLEMENT_COPYTO

#endif
