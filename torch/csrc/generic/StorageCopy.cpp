#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StorageCopy.cpp"
#else

#define IMPLEMENT_COPY_WRAPPER(NAME,TYPEA,TYPEB)                               \
    IMPLEMENT_COPY_WRAPPER_FULLNAME(THStorage_(NAME), TYPEA, TYPEB)

#define IMPLEMENT_COPY_WRAPPER_FULLNAME(NAME,TYPEA,TYPEB)                      \
void TH_CONCAT_2(_THPCopy_,NAME)(PyObject *dst, PyObject *src)                 \
{                                                                              \
  NAME(LIBRARY_STATE ((TYPEA *)dst)->cdata,                                    \
          ((TYPEB *)src)->cdata);                                              \
}

IMPLEMENT_COPY_WRAPPER(copyByte,    THPStorage,          THPByteStorage)
IMPLEMENT_COPY_WRAPPER(copyChar,    THPStorage,          THPCharStorage)
IMPLEMENT_COPY_WRAPPER(copyShort,   THPStorage,          THPShortStorage)
IMPLEMENT_COPY_WRAPPER(copyInt,     THPStorage,          THPIntStorage)
IMPLEMENT_COPY_WRAPPER(copyLong,    THPStorage,          THPLongStorage)
IMPLEMENT_COPY_WRAPPER(copyFloat,   THPStorage,          THPFloatStorage)
IMPLEMENT_COPY_WRAPPER(copyDouble,  THPStorage,          THPDoubleStorage)

#ifdef THC_GENERIC_FILE
IMPLEMENT_COPY_WRAPPER(copyCudaByte,    THCPStorage,     THCPByteStorage)
IMPLEMENT_COPY_WRAPPER(copyCudaChar,    THCPStorage,     THCPCharStorage)
IMPLEMENT_COPY_WRAPPER(copyCudaShort,   THCPStorage,     THCPShortStorage)
IMPLEMENT_COPY_WRAPPER(copyCudaInt,     THCPStorage,     THCPIntStorage)
IMPLEMENT_COPY_WRAPPER(copyCudaLong,    THCPStorage,     THCPLongStorage)
IMPLEMENT_COPY_WRAPPER(copyCudaFloat,   THCPStorage,     THCPFloatStorage)
IMPLEMENT_COPY_WRAPPER(copyCudaDouble,  THCPStorage,     THCPDoubleStorage)
#ifdef CUDA_HALF_TENSOR
IMPLEMENT_COPY_WRAPPER(copyCudaHalf,    THCPStorage,     THCPHalfStorage)
#endif

IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THByteStorage_copyCuda  , Real), THPByteStorage,   THCPStorage);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THCharStorage_copyCuda  , Real), THPCharStorage,   THCPStorage);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THShortStorage_copyCuda , Real), THPShortStorage,  THCPStorage);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THIntStorage_copyCuda   , Real), THPIntStorage,    THCPStorage);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THLongStorage_copyCuda  , Real), THPLongStorage,   THCPStorage);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THFloatStorage_copyCuda , Real), THPFloatStorage,  THCPStorage);
IMPLEMENT_COPY_WRAPPER_FULLNAME(TH_CONCAT_2(THDoubleStorage_copyCuda, Real), THPDoubleStorage, THCPStorage);

#endif

#endif
