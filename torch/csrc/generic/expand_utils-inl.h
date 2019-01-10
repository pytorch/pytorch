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
  void expand(LIBRARY_STATE_TYPE TH##TYPEC##Tensor *r, TH##TYPEC##Tensor *tensor,  \
              THLongStorage *sizes) {                                              \
    TH##TYPEC##Tensor_expand(THC_STATE r, tensor, sizes);                          \
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
