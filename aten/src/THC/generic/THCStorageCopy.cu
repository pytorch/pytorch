#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorageCopy.cu"
#else

// conversions are delegated to THCTensor implementation
#define THC_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC,TYPECUDA)                                 \
void THCStorage_(copyCuda##TYPEC)(THCState *state, THCStorage *self, struct THCuda##TYPECUDA##Storage *src)  \
{                                                                                       \
  THArgCheck(self->numel() == src->numel(), 2, "size does not match");                        \
  THCTensor* selfTensor = THCTensor_(newWithStorage1d)(state, self, 0, self->numel(), 1);  \
  struct THCuda##TYPECUDA##Tensor* srcTensor =                                          \
      THCuda##TYPECUDA##Tensor_newWithStorage1d(state, src, 0, src->numel(), 1);           \
  THCTensor_(copy)(state, selfTensor, srcTensor);                            \
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
THC_CUDA_STORAGE_IMPLEMENT_COPY(Half,Half)
THC_CUDA_STORAGE_IMPLEMENT_COPY(Bool,Bool)
THC_CUDA_STORAGE_IMPLEMENT_COPY(BFloat16,BFloat16)

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
