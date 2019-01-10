#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.cu"
#else

void THCStorage_(rawCopy)(THCState *state, THCStorage *self, real *src)
{
  THCudaCheck(cudaMemcpyAsync(self->data, src, self->size * sizeof(real), cudaMemcpyDeviceToDevice, THCState_getCurrentStream(state)));
}

// conversions are delegated to THCTensor implementation
#define THC_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC,TYPECUDA)                                 \
void THCStorage_(copyCuda##TYPEC)(THCState *state, THCStorage *self, struct THCuda##TYPECUDA##Storage *src)  \
{                                                                                       \
  THArgCheck(self->size == src->size, 2, "size does not match");                        \
  THCTensor* selfTensor = THCTensor_(newWithStorage1d)(state, self, 0, self->size, 1);  \
  struct THCuda##TYPECUDA##Tensor* srcTensor =                                          \
      THCuda##TYPECUDA##Tensor_newWithStorage1d(state, src, 0, src->size, 1);           \
  THCTensor_(copyCuda##TYPEC)(state, selfTensor, srcTensor);                            \
  THCuda##TYPECUDA##Tensor_free(state, srcTensor);                                      \
  THCTensor_(free)(state, selfTensor);                                                  \
}

THC_CUDA_STORAGE_IMPLEMENT_COPY(Byte,Byte)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Char,Char)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Short,Short)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Int,Int)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Long,Long)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Float,)  // i.e. float
THC_CUDA_STORAGE_IMPLEMENT_COPY(Double,Double)
#ifdef CUDA_HALF_TENSOR
THC_CUDA_STORAGE_IMPLEMENT_COPY(Half,Half)
#endif

#undef THC_CUDA_STORAGE_IMPLEMENT_COPY

void THCStorage_(copyCuda)(THCState *state, THCStorage *self, THCStorage *src)
{
  THCStorage_(TH_CONCAT_2(copyCuda, Real))(state, self, src);
}

void THCStorage_(copy)(THCState *state, THCStorage *self, THCStorage *src)
{
  THCStorage_(copyCuda)(state, self, src);
}

#endif
