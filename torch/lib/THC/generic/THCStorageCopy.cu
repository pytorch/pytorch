#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.cu"
#else

void THCStorage_(rawCopy)(THCState *state, THCStorage *self, real *src)
{
  THCudaCheck(cudaMemcpyAsync(self->data, src, self->size * sizeof(real), cudaMemcpyDeviceToDevice, THCState_getCurrentStream(state)));
}

void THCStorage_(copy)(THCState *state, THCStorage *self, THCStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(real), cudaMemcpyDeviceToDevice, THCState_getCurrentStream(state)));
}

void THCStorage_(copyCuda)(THCState *state, THCStorage *self, THCStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THCudaCheck(cudaMemcpyAsync(self->data, src->data, self->size * sizeof(real), cudaMemcpyDeviceToDevice, THCState_getCurrentStream(state)));
}

// conversions are mediated by the CPU
// yes, this is slow; feel free to write CUDA kernels for this
#ifndef THC_REAL_IS_HALF
#define THC_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC,TYPECUDA)                            \
  void THCStorage_(copyCuda##TYPEC)(THCState *state, THCStorage *self, struct THCuda##TYPECUDA##Storage *src)  \
  {                                                                      \
    if(THCTypeIdx_(Real) == THCTypeIdx_(TYPEC)) {                        \
      THCStorage_(copy)(state, self, (THCStorage*) src);   /* cast just removes compiler warning */ \
    } else {                                                             \
      THArgCheck(self->size == src->size, 2, "size does not match");     \
      TH##TYPEC##Storage *buffer1 = TH##TYPEC##Storage_newWithSize(src->size); \
      THStorage *buffer2  = THStorage_(newWithSize)(src->size);          \
      TH##TYPEC##Storage_copyCuda(state, buffer1, src);                  \
      THStorage_(copy##TYPEC)(buffer2, buffer1);                         \
      THCStorage_(copyCPU)(state, self, buffer2);                        \
      TH##TYPEC##Storage_free(buffer1);                                  \
      THStorage_(free)(buffer2);                                         \
    }                                                                    \
  }
#else
#define THC_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC,TYPECUDA)                            \
  void THCStorage_(copyCuda##TYPEC)(THCState *state, THCStorage *self, struct THCuda##TYPECUDA##Storage *src)  \
  {                                                                      \
    THArgCheck(self->size == src->size, 2, "size does not match");       \
    if(THCTypeIdx_(TYPEC) == THCTypeIdxFloat) {                          \
      THCFloat2Half(state, self->data, (float*) src->data, src->size);   /* cast removes compiler error */     \
    } else {                                                             \
      THCudaStorage *buffer = THCudaStorage_newWithSize(state, src->size); \
      THCudaStorage_copyCuda##TYPEC(state, buffer, src);                 \
      THCFloat2Half(state, self->data, buffer->data, buffer->size);      \
      THCudaStorage_free(state, buffer);                                 \
    }                                                                    \
  }
#endif

THC_CUDA_STORAGE_IMPLEMENT_COPY(Byte,Byte)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Char,Char)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Short,Short)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Int,Int)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Long,Long)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Float,)  // i.e. float
THC_CUDA_STORAGE_IMPLEMENT_COPY(Double,Double)

#ifdef CUDA_HALF_TENSOR
#define FLOAT_COPY(TYPE) TH_CONCAT_3(TH, CReal, Storage_copyCudaFloat)
void THCStorage_(copyCudaHalf)(THCState *state, THCStorage *self, struct THCudaHalfStorage *src)
{
    if(THCTypeIdx_(Real) == THCTypeIdxHalf) {
      THCStorage_(copy)(state, self, (THCStorage*) src);   /* cast just removes compiler warning */
    } else {
        THArgCheck(self->size == src->size, 2, "size does not match");
        THCudaStorage *buffer = THCudaStorage_newWithSize(state, src->size);
        THCHalf2Float(state, buffer->data, src->data, src->size);
        FLOAT_COPY(Real)(state, self, buffer);
        THCudaStorage_free(state, buffer);
    }
}
#undef FLOAT_COPY
#endif // CUDA_HALF_TENSOR

#undef THC_CUDA_STORAGE_IMPLEMENT_COPY

#endif
