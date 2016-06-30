#define DECLARE_COPY(THNAME)                                                   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyDouble)(PyObject *dst, PyObject *src);  \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyFloat)(PyObject *dst, PyObject *src);   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyLong)(PyObject *dst, PyObject *src);    \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyInt)(PyObject *dst, PyObject *src);     \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyShort)(PyObject *dst, PyObject *src);   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyChar)(PyObject *dst, PyObject *src);    \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyByte)(PyObject *dst, PyObject *src);
DECLARE_COPY(THDoubleTensor)
DECLARE_COPY(THFloatTensor)
DECLARE_COPY(THLongTensor)
DECLARE_COPY(THIntTensor)
DECLARE_COPY(THShortTensor)
DECLARE_COPY(THCharTensor)
DECLARE_COPY(THByteTensor)

DECLARE_COPY(THDoubleStorage)
DECLARE_COPY(THFloatStorage)
DECLARE_COPY(THLongStorage)
DECLARE_COPY(THIntStorage)
DECLARE_COPY(THShortStorage)
DECLARE_COPY(THCharStorage)
DECLARE_COPY(THByteStorage)
#undef DECLARE_COPY

static bool THPModule_initCopy(PyObject *unused)
{
#define INIT_TENSOR_COPY(TYPE, THNAME)                                         \
  tensor_copy_handlers.insert({{TYPE, THPDoubleTensorClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyDouble)});  \
  tensor_copy_handlers.insert({{TYPE, THPFloatTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyFloat)});   \
  tensor_copy_handlers.insert({{TYPE, THPLongTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyLong)});    \
  tensor_copy_handlers.insert({{TYPE, THPIntTensorClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyInt)});     \
  tensor_copy_handlers.insert({{TYPE, THPShortTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyShort)});   \
  tensor_copy_handlers.insert({{TYPE, THPCharTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyChar)});    \
  tensor_copy_handlers.insert({{TYPE, THPByteTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyByte)});

#define INIT_STORAGE_COPY(TYPE, THNAME)                                        \
  storage_copy_handlers.insert({{TYPE, THPDoubleStorageClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyDouble)});  \
  storage_copy_handlers.insert({{TYPE, THPFloatStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyFloat)});   \
  storage_copy_handlers.insert({{TYPE, THPLongStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyLong)});    \
  storage_copy_handlers.insert({{TYPE, THPIntStorageClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyInt)});     \
  storage_copy_handlers.insert({{TYPE, THPShortStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyShort)});   \
  storage_copy_handlers.insert({{TYPE, THPCharStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyChar)});    \
  storage_copy_handlers.insert({{TYPE, THPByteStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyByte)});

  INIT_TENSOR_COPY(THPDoubleTensorClass, THDoubleTensor);
  INIT_TENSOR_COPY(THPFloatTensorClass, THFloatTensor);
  INIT_TENSOR_COPY(THPLongTensorClass, THLongTensor);
  INIT_TENSOR_COPY(THPIntTensorClass, THIntTensor);
  INIT_TENSOR_COPY(THPShortTensorClass, THShortTensor);
  INIT_TENSOR_COPY(THPCharTensorClass, THCharTensor);
  INIT_TENSOR_COPY(THPByteTensorClass, THByteTensor);

  INIT_STORAGE_COPY(THPDoubleStorageClass,  THDoubleStorage);
  INIT_STORAGE_COPY(THPFloatStorageClass,   THFloatStorage);
  INIT_STORAGE_COPY(THPLongStorageClass,    THLongStorage);
  INIT_STORAGE_COPY(THPIntStorageClass,     THIntStorage);
  INIT_STORAGE_COPY(THPShortStorageClass,   THShortStorage);
  INIT_STORAGE_COPY(THPCharStorageClass,    THCharStorage);
  INIT_STORAGE_COPY(THPByteStorageClass,    THByteStorage);

  return true;
#undef INIT_TENSOR_COPY
#undef INIT_STORAGE_COPY
}

