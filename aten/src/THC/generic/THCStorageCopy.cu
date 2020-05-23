#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorageCopy.cu"
#else

// conversions are delegated to THCTensor implementation
#define THC_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC, TYPECUDA)              \
  void THCStorage_(copyCuda##TYPEC)(                                  \
      THCState * state,                                               \
      THCStorage * self,                                              \
      struct THCuda##TYPECUDA##Storage * src) {                       \
    size_t self_numel = self->nbytes() / sizeof(scalar_t);            \
    size_t src_numel =                                                \
        src->nbytes() / THCuda##TYPECUDA##Storage_elementSize(state); \
    THArgCheck(self_numel == src_numel, 2, "size does not match");    \
    THCTensor* selfTensor =                                           \
        THCTensor_(newWithStorage1d)(state, self, 0, self_numel, 1);  \
    struct THCuda##TYPECUDA##Tensor* srcTensor =                      \
        THCuda##TYPECUDA##Tensor_newWithStorage1d(                    \
            state, src, 0, src_numel, 1);                             \
    THCTensor_(copy)(state, selfTensor, srcTensor);                   \
    THCuda##TYPECUDA##Tensor_free(state, srcTensor);                  \
    THCTensor_(free)(state, selfTensor);                              \
  }

#if !defined(THC_REAL_IS_COMPLEXFLOAT) && !defined(THC_REAL_IS_COMPLEXDOUBLE)
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
#else
  THC_CUDA_STORAGE_IMPLEMENT_COPY(ComplexFloat,ComplexFloat)
  THC_CUDA_STORAGE_IMPLEMENT_COPY(ComplexDouble,ComplexDouble)
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
