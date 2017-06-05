// We need separate THC_STATE from LIBRARY_STATE because this is called for copy functions.
// Copy functions have a THCState if either the source or destination are GPU types,
// but expand functions don't have this property.  So, if we have a source CPU type
// and destination GPU type, we will have a THCState but shouldn't pass it to the expand
// function.

#define IMPLEMENT_NEWFOREXPAND(TYPEC, THC_STATE_NOARGS)         \
  template<>                                                    \
  TH##TYPEC##Tensor *newForExpand(LIBRARY_STATE_TYPE_NOARGS) {  \
    return TH##TYPEC##Tensor_new(THC_STATE_NOARGS);             \
  }

IMPLEMENT_NEWFOREXPAND(Byte,)
IMPLEMENT_NEWFOREXPAND(Char,)
IMPLEMENT_NEWFOREXPAND(Short,)
IMPLEMENT_NEWFOREXPAND(Int,)
IMPLEMENT_NEWFOREXPAND(Long,)
IMPLEMENT_NEWFOREXPAND(Float,)
IMPLEMENT_NEWFOREXPAND(Double,)
IMPLEMENT_NEWFOREXPAND(Half,)

#ifdef CUDA_EXPAND
IMPLEMENT_NEWFOREXPAND(CudaByte, LIBRARY_STATE_NOARGS)
IMPLEMENT_NEWFOREXPAND(CudaChar, LIBRARY_STATE_NOARGS)
IMPLEMENT_NEWFOREXPAND(CudaShort, LIBRARY_STATE_NOARGS)
IMPLEMENT_NEWFOREXPAND(CudaInt, LIBRARY_STATE_NOARGS)
IMPLEMENT_NEWFOREXPAND(CudaLong, LIBRARY_STATE_NOARGS)
IMPLEMENT_NEWFOREXPAND(Cuda, LIBRARY_STATE_NOARGS)
IMPLEMENT_NEWFOREXPAND(CudaDouble, LIBRARY_STATE_NOARGS)
#ifdef CUDA_HALF_TENSOR
IMPLEMENT_NEWFOREXPAND(CudaHalf, LIBRARY_STATE_NOARGS)
#endif
#endif

#undef IMPLEMENT_NEWFOREXPAND

#define IMPLEMENT_EXPAND(TYPEC, THC_STATE)                                         \
  template<>                                                                       \
  int expand(LIBRARY_STATE_TYPE TH##TYPEC##Tensor *r, TH##TYPEC##Tensor *tensor,   \
             THLongStorage *sizes, int raiseErrors) {                              \
    return TH##TYPEC##Tensor_expand(THC_STATE r, tensor, sizes, raiseErrors);  \
  }

IMPLEMENT_EXPAND(Byte,)
IMPLEMENT_EXPAND(Char,)
IMPLEMENT_EXPAND(Short,)
IMPLEMENT_EXPAND(Int,)
IMPLEMENT_EXPAND(Long,)
IMPLEMENT_EXPAND(Float,)
IMPLEMENT_EXPAND(Double,)
IMPLEMENT_EXPAND(Half,)

#ifdef CUDA_EXPAND
IMPLEMENT_EXPAND(CudaByte, LIBRARY_STATE)
IMPLEMENT_EXPAND(CudaChar, LIBRARY_STATE)
IMPLEMENT_EXPAND(CudaShort, LIBRARY_STATE)
IMPLEMENT_EXPAND(CudaInt, LIBRARY_STATE)
IMPLEMENT_EXPAND(CudaLong, LIBRARY_STATE)
IMPLEMENT_EXPAND(Cuda, LIBRARY_STATE)
IMPLEMENT_EXPAND(CudaDouble, LIBRARY_STATE)
#ifdef CUDA_HALF_TENSOR
IMPLEMENT_EXPAND(CudaHalf, LIBRARY_STATE)
#endif
#endif

#undef IMPLEMENT_EXPAND

#define IMPLEMENT_EXPAND2(TYPEC, THC_STATE)                                       \
  template <>                                                                     \
  int expand2(LIBRARY_STATE_TYPE TH##TYPEC##Tensor *r1, TH##TYPEC##Tensor *r2,    \
              TH##TYPEC##Tensor *e1, TH##TYPEC##Tensor *e2, int raiseErrors) {    \
    return TH##TYPEC##Tensor_expand2(THC_STATE r1, r2, e1, e2, raiseErrors);      \
}

IMPLEMENT_EXPAND2(Byte,)
IMPLEMENT_EXPAND2(Char,)
IMPLEMENT_EXPAND2(Short,)
IMPLEMENT_EXPAND2(Int,)
IMPLEMENT_EXPAND2(Long,)
IMPLEMENT_EXPAND2(Float,)
IMPLEMENT_EXPAND2(Double,)
IMPLEMENT_EXPAND2(Half,)

#ifdef CUDA_EXPAND
IMPLEMENT_EXPAND2(CudaByte, LIBRARY_STATE)
IMPLEMENT_EXPAND2(CudaChar, LIBRARY_STATE)
IMPLEMENT_EXPAND2(CudaShort, LIBRARY_STATE)
IMPLEMENT_EXPAND2(CudaInt, LIBRARY_STATE)
IMPLEMENT_EXPAND2(CudaLong, LIBRARY_STATE)
IMPLEMENT_EXPAND2(Cuda, LIBRARY_STATE)
IMPLEMENT_EXPAND2(CudaDouble, LIBRARY_STATE)
#ifdef CUDA_HALF_TENSOR
IMPLEMENT_EXPAND2(CudaHalf, LIBRARY_STATE)
#endif
#endif

#undef IMPLEMENT_EXPAND2

#define IMPLEMENT_EXPAND3(TYPEC, THC_STATE)                                                            \
  template <>                                                                                          \
  int expand3(LIBRARY_STATE_TYPE TH##TYPEC##Tensor *r1, TH##TYPEC##Tensor *r2, TH##TYPEC##Tensor *r3,  \
              TH##TYPEC##Tensor *e1, TH##TYPEC##Tensor *e2, TH##TYPEC##Tensor *e3, int raiseErrors) {  \
  return TH##TYPEC##Tensor_expand3(THC_STATE r1, r2, r3, e1, e2, e3, raiseErrors);                     \
}

IMPLEMENT_EXPAND3(Byte,)
IMPLEMENT_EXPAND3(Char,)
IMPLEMENT_EXPAND3(Short,)
IMPLEMENT_EXPAND3(Int,)
IMPLEMENT_EXPAND3(Long,)
IMPLEMENT_EXPAND3(Float,)
IMPLEMENT_EXPAND3(Double,)
IMPLEMENT_EXPAND3(Half,)

#ifdef CUDA_EXPAND
IMPLEMENT_EXPAND3(CudaByte, LIBRARY_STATE)
IMPLEMENT_EXPAND3(CudaChar, LIBRARY_STATE)
IMPLEMENT_EXPAND3(CudaShort, LIBRARY_STATE)
IMPLEMENT_EXPAND3(CudaInt, LIBRARY_STATE)
IMPLEMENT_EXPAND3(CudaLong, LIBRARY_STATE)
IMPLEMENT_EXPAND3(Cuda, LIBRARY_STATE)
IMPLEMENT_EXPAND3(CudaDouble, LIBRARY_STATE)
#ifdef CUDA_HALF_TENSOR
IMPLEMENT_EXPAND3(CudaHalf, LIBRARY_STATE)
#endif
#endif

#undef IMPLEMENT_EXPAND3
