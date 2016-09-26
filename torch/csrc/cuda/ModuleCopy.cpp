#include "../ModuleCopy.h"

#define DECLARE_COPY(THNAME)                                                   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyDouble)(PyObject *dst, PyObject *src);  \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyFloat)(PyObject *dst, PyObject *src);   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyLong)(PyObject *dst, PyObject *src);    \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyInt)(PyObject *dst, PyObject *src);     \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyShort)(PyObject *dst, PyObject *src);   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyChar)(PyObject *dst, PyObject *src);    \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyByte)(PyObject *dst, PyObject *src);

#define DECLARE_CUDA_COPY(THNAME)                                              \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaDouble)(PyObject *dst, PyObject *src);  \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaFloat)(PyObject *dst, PyObject *src);   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaLong)(PyObject *dst, PyObject *src);    \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaInt)(PyObject *dst, PyObject *src);     \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaShort)(PyObject *dst, PyObject *src);   \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaChar)(PyObject *dst, PyObject *src);    \
void TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaByte)(PyObject *dst, PyObject *src);

DECLARE_CUDA_COPY(THDoubleTensor)
DECLARE_CUDA_COPY(THFloatTensor)
DECLARE_CUDA_COPY(THLongTensor)
DECLARE_CUDA_COPY(THIntTensor)
DECLARE_CUDA_COPY(THShortTensor)
DECLARE_CUDA_COPY(THCharTensor)
DECLARE_CUDA_COPY(THByteTensor)

DECLARE_COPY(THCudaDoubleTensor)
DECLARE_COPY(THCudaTensor)
DECLARE_COPY(THCudaLongTensor)
DECLARE_COPY(THCudaIntTensor)
DECLARE_COPY(THCudaShortTensor)
DECLARE_COPY(THCudaCharTensor)
DECLARE_COPY(THCudaByteTensor)

DECLARE_CUDA_COPY(THCudaDoubleTensor)
DECLARE_CUDA_COPY(THCudaTensor)
DECLARE_CUDA_COPY(THCudaLongTensor)
DECLARE_CUDA_COPY(THCudaIntTensor)
DECLARE_CUDA_COPY(THCudaShortTensor)
DECLARE_CUDA_COPY(THCudaCharTensor)
DECLARE_CUDA_COPY(THCudaByteTensor)

DECLARE_CUDA_COPY(THDoubleStorage)
DECLARE_CUDA_COPY(THFloatStorage)
DECLARE_CUDA_COPY(THLongStorage)
DECLARE_CUDA_COPY(THIntStorage)
DECLARE_CUDA_COPY(THShortStorage)
DECLARE_CUDA_COPY(THCharStorage)
DECLARE_CUDA_COPY(THByteStorage)

DECLARE_COPY(THCudaDoubleStorage)
DECLARE_COPY(THCudaStorage)
DECLARE_COPY(THCudaLongStorage)
DECLARE_COPY(THCudaIntStorage)
DECLARE_COPY(THCudaShortStorage)
DECLARE_COPY(THCudaCharStorage)
DECLARE_COPY(THCudaByteStorage)

DECLARE_CUDA_COPY(THCudaDoubleStorage)
DECLARE_CUDA_COPY(THCudaStorage)
DECLARE_CUDA_COPY(THCudaLongStorage)
DECLARE_CUDA_COPY(THCudaIntStorage)
DECLARE_CUDA_COPY(THCudaShortStorage)
DECLARE_CUDA_COPY(THCudaCharStorage)
DECLARE_CUDA_COPY(THCudaByteStorage)
#undef DECLARE_COPY

#define DECLARE_ASYNC_COPY(TYPE)                                               \
void TH_CONCAT_3(THCP,TYPE,Tensor_copyAsyncCPU)(PyObject *dst, PyObject *src); \
void TH_CONCAT_3(THP,TYPE,Tensor_copyAsyncGPU)(PyObject *dst, PyObject *src);

DECLARE_ASYNC_COPY(Double)
DECLARE_ASYNC_COPY(Float)
DECLARE_ASYNC_COPY(Long)
DECLARE_ASYNC_COPY(Int)
DECLARE_ASYNC_COPY(Short)
DECLARE_ASYNC_COPY(Char)
DECLARE_ASYNC_COPY(Byte)
#undef DECLARE_ASYNC_COPY

extern PyObject *THPDoubleStorageClass;
extern PyObject *THPFloatStorageClass;
extern PyObject *THPLongStorageClass;
extern PyObject *THPIntStorageClass;
extern PyObject *THPShortStorageClass;
extern PyObject *THPCharStorageClass;
extern PyObject *THPByteStorageClass;

extern PyObject *THPDoubleTensorClass;
extern PyObject *THPFloatTensorClass;
extern PyObject *THPLongTensorClass;
extern PyObject *THPIntTensorClass;
extern PyObject *THPShortTensorClass;
extern PyObject *THPCharTensorClass;
extern PyObject *THPByteTensorClass;

static bool THCPModule_initCopy()
{
// TODO: half
#define INIT_TENSOR_GPU_CPU_COPY(TYPE, THNAME)                                        \
  tensor_copy_handlers.insert({{TYPE, THCPDoubleTensorClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaDouble)});  \
  tensor_copy_handlers.insert({{TYPE, THCPFloatTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaFloat)});   \
  tensor_copy_handlers.insert({{TYPE, THCPLongTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaLong)});    \
  tensor_copy_handlers.insert({{TYPE, THCPIntTensorClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaInt)});     \
  tensor_copy_handlers.insert({{TYPE, THCPShortTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaShort)});   \
  tensor_copy_handlers.insert({{TYPE, THCPCharTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaChar)});    \
  tensor_copy_handlers.insert({{TYPE, THCPByteTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaByte)});

#define INIT_TENSOR_GPU_GPU_COPY(TYPE, THNAME)                                        \
  tensor_copy_handlers.insert({{TYPE, THCPDoubleTensorClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaDouble)});  \
  tensor_copy_handlers.insert({{TYPE, THCPFloatTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaFloat)});   \
  tensor_copy_handlers.insert({{TYPE, THCPLongTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaLong)});    \
  tensor_copy_handlers.insert({{TYPE, THCPIntTensorClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaInt)});     \
  tensor_copy_handlers.insert({{TYPE, THCPShortTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaShort)});   \
  tensor_copy_handlers.insert({{TYPE, THCPCharTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaChar)});    \
  tensor_copy_handlers.insert({{TYPE, THCPByteTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaByte)});    \
  /* CUDA copy launches are always async */                                                                      \
  tensor_async_copy_handlers.insert({{TYPE, THCPDoubleTensorClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaDouble)}); \
  tensor_async_copy_handlers.insert({{TYPE, THCPFloatTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaFloat)});  \
  tensor_async_copy_handlers.insert({{TYPE, THCPLongTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaLong)});   \
  tensor_async_copy_handlers.insert({{TYPE, THCPIntTensorClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaInt)});    \
  tensor_async_copy_handlers.insert({{TYPE, THCPShortTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaShort)});  \
  tensor_async_copy_handlers.insert({{TYPE, THCPCharTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaChar)});   \
  tensor_async_copy_handlers.insert({{TYPE, THCPByteTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaByte)});

#define INIT_TENSOR_CPU_GPU_COPY(TYPE, THNAME)                                        \
  tensor_copy_handlers.insert({{TYPE, THPDoubleTensorClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyDouble)});  \
  tensor_copy_handlers.insert({{TYPE, THPFloatTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyFloat)});   \
  tensor_copy_handlers.insert({{TYPE, THPLongTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyLong)});    \
  tensor_copy_handlers.insert({{TYPE, THPIntTensorClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyInt)});     \
  tensor_copy_handlers.insert({{TYPE, THPShortTensorClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyShort)});   \
  tensor_copy_handlers.insert({{TYPE, THPCharTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyChar)});    \
  tensor_copy_handlers.insert({{TYPE, THPByteTensorClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyByte)});

#define INIT_TENSOR_ASYNC_COPY(TYPE)                                           \
  tensor_async_copy_handlers.insert({{TH_CONCAT_3(THP,TYPE,TensorClass), TH_CONCAT_3(THCP,TYPE,TensorClass)}, TH_CONCAT_3(THP,TYPE,Tensor_copyAsyncGPU)}); \
  tensor_async_copy_handlers.insert({{TH_CONCAT_3(THCP,TYPE,TensorClass), TH_CONCAT_3(THP,TYPE,TensorClass)}, TH_CONCAT_3(THCP,TYPE,Tensor_copyAsyncCPU)});

  INIT_TENSOR_GPU_CPU_COPY(THPDoubleTensorClass, THDoubleTensor);
  INIT_TENSOR_GPU_CPU_COPY(THPFloatTensorClass,  THFloatTensor);
  INIT_TENSOR_GPU_CPU_COPY(THPLongTensorClass,   THLongTensor);
  INIT_TENSOR_GPU_CPU_COPY(THPIntTensorClass,    THIntTensor);
  INIT_TENSOR_GPU_CPU_COPY(THPShortTensorClass,  THShortTensor);
  INIT_TENSOR_GPU_CPU_COPY(THPCharTensorClass,   THCharTensor);
  INIT_TENSOR_GPU_CPU_COPY(THPByteTensorClass,   THByteTensor);

  INIT_TENSOR_GPU_GPU_COPY(THCPDoubleTensorClass, THCudaDoubleTensor);
  INIT_TENSOR_GPU_GPU_COPY(THCPFloatTensorClass,  THCudaTensor);
  INIT_TENSOR_GPU_GPU_COPY(THCPLongTensorClass,   THCudaLongTensor);
  INIT_TENSOR_GPU_GPU_COPY(THCPIntTensorClass,    THCudaIntTensor);
  INIT_TENSOR_GPU_GPU_COPY(THCPShortTensorClass,  THCudaShortTensor);
  INIT_TENSOR_GPU_GPU_COPY(THCPCharTensorClass,   THCudaCharTensor);
  INIT_TENSOR_GPU_GPU_COPY(THCPByteTensorClass,   THCudaByteTensor);

  INIT_TENSOR_CPU_GPU_COPY(THCPDoubleTensorClass, THCudaDoubleTensor);
  INIT_TENSOR_CPU_GPU_COPY(THCPFloatTensorClass,  THCudaTensor);
  INIT_TENSOR_CPU_GPU_COPY(THCPLongTensorClass,   THCudaLongTensor);
  INIT_TENSOR_CPU_GPU_COPY(THCPIntTensorClass,    THCudaIntTensor);
  INIT_TENSOR_CPU_GPU_COPY(THCPShortTensorClass,  THCudaShortTensor);
  INIT_TENSOR_CPU_GPU_COPY(THCPCharTensorClass,   THCudaCharTensor);
  INIT_TENSOR_CPU_GPU_COPY(THCPByteTensorClass,   THCudaByteTensor);

  INIT_TENSOR_ASYNC_COPY(Double)
  INIT_TENSOR_ASYNC_COPY(Float)
  INIT_TENSOR_ASYNC_COPY(Long)
  INIT_TENSOR_ASYNC_COPY(Int)
  INIT_TENSOR_ASYNC_COPY(Short)
  INIT_TENSOR_ASYNC_COPY(Char)
  INIT_TENSOR_ASYNC_COPY(Byte)

#define INIT_STORAGE_GPU_CPU_COPY(TYPE, THNAME)                                \
  storage_copy_handlers.insert({{TYPE, THCPDoubleStorageClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaDouble)});  \
  storage_copy_handlers.insert({{TYPE, THCPFloatStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaFloat)});   \
  storage_copy_handlers.insert({{TYPE, THCPLongStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaLong)});    \
  storage_copy_handlers.insert({{TYPE, THCPIntStorageClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaInt)});     \
  storage_copy_handlers.insert({{TYPE, THCPShortStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaShort)});   \
  storage_copy_handlers.insert({{TYPE, THCPCharStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaChar)});    \
  storage_copy_handlers.insert({{TYPE, THCPByteStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaByte)});

#define INIT_STORAGE_GPU_GPU_COPY(TYPE, THNAME)                                \
  storage_copy_handlers.insert({{TYPE, THCPDoubleStorageClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaDouble)});  \
  storage_copy_handlers.insert({{TYPE, THCPFloatStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaFloat)});   \
  storage_copy_handlers.insert({{TYPE, THCPLongStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaLong)});    \
  storage_copy_handlers.insert({{TYPE, THCPIntStorageClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaInt)});     \
  storage_copy_handlers.insert({{TYPE, THCPShortStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaShort)});   \
  storage_copy_handlers.insert({{TYPE, THCPCharStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaChar)});    \
  storage_copy_handlers.insert({{TYPE, THCPByteStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyCudaByte)});

#define INIT_STORAGE_CPU_GPU_COPY(TYPE, THNAME)                                \
  storage_copy_handlers.insert({{TYPE, THPDoubleStorageClass},  TH_CONCAT_3(_THPCopy_,THNAME,_copyDouble)});  \
  storage_copy_handlers.insert({{TYPE, THPFloatStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyFloat)});   \
  storage_copy_handlers.insert({{TYPE, THPLongStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyLong)});    \
  storage_copy_handlers.insert({{TYPE, THPIntStorageClass},     TH_CONCAT_3(_THPCopy_,THNAME,_copyInt)});     \
  storage_copy_handlers.insert({{TYPE, THPShortStorageClass},   TH_CONCAT_3(_THPCopy_,THNAME,_copyShort)});   \
  storage_copy_handlers.insert({{TYPE, THPCharStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyChar)});    \
  storage_copy_handlers.insert({{TYPE, THPByteStorageClass},    TH_CONCAT_3(_THPCopy_,THNAME,_copyByte)});

  INIT_STORAGE_GPU_CPU_COPY(THPDoubleStorageClass, THDoubleStorage);
  INIT_STORAGE_GPU_CPU_COPY(THPFloatStorageClass,  THFloatStorage);
  INIT_STORAGE_GPU_CPU_COPY(THPLongStorageClass,   THLongStorage);
  INIT_STORAGE_GPU_CPU_COPY(THPIntStorageClass,    THIntStorage);
  INIT_STORAGE_GPU_CPU_COPY(THPShortStorageClass,  THShortStorage);
  INIT_STORAGE_GPU_CPU_COPY(THPCharStorageClass,   THCharStorage);
  INIT_STORAGE_GPU_CPU_COPY(THPByteStorageClass,   THByteStorage);

  INIT_STORAGE_GPU_GPU_COPY(THCPDoubleStorageClass, THCudaDoubleStorage);
  INIT_STORAGE_GPU_GPU_COPY(THCPFloatStorageClass,  THCudaStorage);
  INIT_STORAGE_GPU_GPU_COPY(THCPLongStorageClass,   THCudaLongStorage);
  INIT_STORAGE_GPU_GPU_COPY(THCPIntStorageClass,    THCudaIntStorage);
  INIT_STORAGE_GPU_GPU_COPY(THCPShortStorageClass,  THCudaShortStorage);
  INIT_STORAGE_GPU_GPU_COPY(THCPCharStorageClass,   THCudaCharStorage);
  INIT_STORAGE_GPU_GPU_COPY(THCPByteStorageClass,   THCudaByteStorage);

  INIT_STORAGE_CPU_GPU_COPY(THCPDoubleStorageClass, THCudaDoubleStorage);
  INIT_STORAGE_CPU_GPU_COPY(THCPFloatStorageClass,  THCudaStorage);
  INIT_STORAGE_CPU_GPU_COPY(THCPLongStorageClass,   THCudaLongStorage);
  INIT_STORAGE_CPU_GPU_COPY(THCPIntStorageClass,    THCudaIntStorage);
  INIT_STORAGE_CPU_GPU_COPY(THCPShortStorageClass,  THCudaShortStorage);
  INIT_STORAGE_CPU_GPU_COPY(THCPCharStorageClass,   THCudaCharStorage);
  INIT_STORAGE_CPU_GPU_COPY(THCPByteStorageClass,   THCudaByteStorage);

  return true;
#undef INIT_TENSOR_GPU_CPU_COPY
#undef INIT_TENSOR_GPU_GPU_COPY
#undef INIT_TENSOR_CPU_GPU_COPY
#undef INIT_TENSOR_ASYNC_COPY
#undef INIT_STORAGE_GPU_CPU_COPY
#undef INIT_STORAGE_GPU_GPU_COPY
#undef INIT_STORAGE_CPU_GPU_COPY
}
