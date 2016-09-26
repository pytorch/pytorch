#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorCopy.cpp"
#else

#define IMPLEMENT_COPY_WRAPPER(NAME,TYPEA,TYPEB)                               \
    IMPLEMENT_COPY_WRAPPER_FULLNAME(THTensor_(NAME), TYPEA, TYPEB)

#define IMPLEMENT_COPY_WRAPPER_FULLNAME(NAME,TYPEA,TYPEB)                      \
void TH_CONCAT_2(_THPCopy_,NAME)(PyObject *dst, PyObject *src)                 \
{                                                                              \
  NAME(LIBRARY_STATE ((TYPEA *)dst)->cdata,                                    \
          ((TYPEB *)src)->cdata);                                              \
}

IMPLEMENT_COPY_WRAPPER(copyByte,    THPTensor,          THPByteTensor)
IMPLEMENT_COPY_WRAPPER(copyChar,    THPTensor,          THPCharTensor)
IMPLEMENT_COPY_WRAPPER(copyShort,   THPTensor,          THPShortTensor)
IMPLEMENT_COPY_WRAPPER(copyInt,     THPTensor,          THPIntTensor)
IMPLEMENT_COPY_WRAPPER(copyLong,    THPTensor,          THPLongTensor)
IMPLEMENT_COPY_WRAPPER(copyFloat,   THPTensor,          THPFloatTensor)
IMPLEMENT_COPY_WRAPPER(copyDouble,  THPTensor,          THPDoubleTensor)

#ifdef THC_GENERIC_FILE
IMPLEMENT_COPY_WRAPPER(copyCudaByte,    THCPTensor,     THCPByteTensor)
IMPLEMENT_COPY_WRAPPER(copyCudaChar,    THCPTensor,     THCPCharTensor)
IMPLEMENT_COPY_WRAPPER(copyCudaShort,   THCPTensor,     THCPShortTensor)
IMPLEMENT_COPY_WRAPPER(copyCudaInt,     THCPTensor,     THCPIntTensor)
IMPLEMENT_COPY_WRAPPER(copyCudaLong,    THCPTensor,     THCPLongTensor)
IMPLEMENT_COPY_WRAPPER(copyCudaFloat,   THCPTensor,     THCPFloatTensor)
IMPLEMENT_COPY_WRAPPER(copyCudaDouble,  THCPTensor,     THCPDoubleTensor)
#ifdef CUDA_HALF_TENSOR
IMPLEMENT_COPY_WRAPPER(copyCudaHalf,    THCPTensor,     THCPHalfTensor)
#endif

IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THByteTensor_copyCuda  , Real), THPByteTensor,   THCPTensor);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THCharTensor_copyCuda  , Real), THPCharTensor,   THCPTensor);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THShortTensor_copyCuda , Real), THPShortTensor,  THCPTensor);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THIntTensor_copyCuda   , Real), THPIntTensor,    THCPTensor);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THLongTensor_copyCuda  , Real), THPLongTensor,   THCPTensor);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THFloatTensor_copyCuda , Real), THPFloatTensor,  THCPTensor);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THDoubleTensor_copyCuda, Real), THPDoubleTensor, THCPTensor);

#endif

#endif
