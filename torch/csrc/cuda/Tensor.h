#ifndef THCP_TENSOR_INC
#define THCP_TENSOR_INC

#define THCPTensor TH_CONCAT_3(THCP,Real,Tensor)
#define THCPTensorStr TH_CONCAT_STRING_3(torch.cuda.,Real,Tensor)
#define THCPTensorClass TH_CONCAT_3(THCP,Real,TensorClass)
#define THCPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)

#define THCPDoubleTensor_Check(obj)  PyObject_IsInstance(obj, THCPDoubleTensorClass)
#define THCPFloatTensor_Check(obj)   PyObject_IsInstance(obj, THCPFloatTensorClass)
#define THCPHalfTensor_Check(obj)    PyObject_IsInstance(obj, THCPHalfTensorClass)
#define THCPLongTensor_Check(obj)    PyObject_IsInstance(obj, THCPLongTensorClass)
#define THCPIntTensor_Check(obj)     PyObject_IsInstance(obj, THCPIntTensorClass)
#define THCPShortTensor_Check(obj)   PyObject_IsInstance(obj, THCPShortTensorClass)
#define THCPCharTensor_Check(obj)    PyObject_IsInstance(obj, THCPCharTensorClass)
#define THCPByteTensor_Check(obj)    PyObject_IsInstance(obj, THCPByteTensorClass)

#define THCPDoubleTensor_CData(obj)  (obj)->cdata
#define THCPFloatTensor_CData(obj)   (obj)->cdata
#define THCPHalfTensor_CData(obj)    (obj)->cdata
#define THCPLongTensor_CData(obj)    (obj)->cdata
#define THCPIntTensor_CData(obj)     (obj)->cdata
#define THCPShortTensor_CData(obj)   (obj)->cdata
#define THCPCharTensor_CData(obj)    (obj)->cdata
#define THCPByteTensor_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THCPTensorType TH_CONCAT_3(THCP,Real,TensorType)
#define THCPTensorBaseStr TH_CONCAT_STRING_3(Cuda,Real,TensorBase)
#define THCPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THCPTensorStatelessType TH_CONCAT_2(CReal,TensorStatelessType)
#define THCPTensorStateless TH_CONCAT_2(CReal,TensorStateless)
#define THCPTensorStatelessMethods TH_CONCAT_2(CReal,TensorStatelessMethods)
#endif

#define THCSPTensor TH_CONCAT_3(THCSP,Real,Tensor)
#define THCSPTensorStr TH_CONCAT_STRING_3(torch.cuda.sparse.,Real,Tensor)
#define THCSPTensorClass TH_CONCAT_3(THCSP,Real,TensorClass)
#define THCSPTensor_(NAME) TH_CONCAT_4(THCSP,Real,Tensor_,NAME)

#define THCSPDoubleTensor_Check(obj)  PyObject_IsInstance(obj, THCSPDoubleTensorClass)
#define THCSPFloatTensor_Check(obj)   PyObject_IsInstance(obj, THCSPFloatTensorClass)
#define THCSPHalfTensor_Check(obj)    PyObject_IsInstance(obj, THCSPHalfTensorClass)
#define THCSPLongTensor_Check(obj)    PyObject_IsInstance(obj, THCSPLongTensorClass)
#define THCSPIntTensor_Check(obj)     PyObject_IsInstance(obj, THCSPIntTensorClass)
#define THCSPShortTensor_Check(obj)   PyObject_IsInstance(obj, THCSPShortTensorClass)
#define THCSPCharTensor_Check(obj)    PyObject_IsInstance(obj, THCSPCharTensorClass)
#define THCSPByteTensor_Check(obj)    PyObject_IsInstance(obj, THCSPByteTensorClass)

#define THCSPDoubleTensor_CData(obj)  (obj)->cdata
#define THCSPFloatTensor_CData(obj)   (obj)->cdata
#define THCSPHalfTensor_CData(obj)    (obj)->cdata
#define THCSPLongTensor_CData(obj)    (obj)->cdata
#define THCSPIntTensor_CData(obj)     (obj)->cdata
#define THCSPShortTensor_CData(obj)   (obj)->cdata
#define THCSPCharTensor_CData(obj)    (obj)->cdata
#define THCSPByteTensor_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THCSPTensorType TH_CONCAT_3(THCSP,Real,TensorType)
#define THCSPTensorBaseStr TH_CONCAT_STRING_3(CudaSparse,Real,TensorBase)
#define THCSPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THCSPTensorStatelessType TH_CONCAT_3(CudaSparse,Real,TensorStatelessType)
#define THCSPTensorStateless TH_CONCAT_3(CudaSparse,Real,TensorStateless)
#define THCSPTensorStatelessMethods TH_CONCAT_3(CudaSparse,Real,TensorStatelessMethods)
#endif

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Tensor.h"
#include <THC/THCGenerateAllTypes.h>

#endif
