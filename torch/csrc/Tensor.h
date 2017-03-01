#ifndef THP_TENSOR_INC
#define THP_TENSOR_INC

#define THPTensor                   TH_CONCAT_3(THP,Real,Tensor)
#define THPTensorStr                TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define THPTensorClass              TH_CONCAT_3(THP,Real,TensorClass)
#define THPTensor_(NAME)            TH_CONCAT_4(THP,Real,Tensor_,NAME)

#define THPDoubleTensor_Check(obj)  PyObject_IsInstance(obj, THPDoubleTensorClass)
#define THPFloatTensor_Check(obj)   PyObject_IsInstance(obj, THPFloatTensorClass)
#define THPHalfTensor_Check(obj)    PyObject_IsInstance(obj, THPHalfTensorClass)
#define THPLongTensor_Check(obj)    PyObject_IsInstance(obj, THPLongTensorClass)
#define THPIntTensor_Check(obj)     PyObject_IsInstance(obj, THPIntTensorClass)
#define THPShortTensor_Check(obj)   PyObject_IsInstance(obj, THPShortTensorClass)
#define THPCharTensor_Check(obj)    PyObject_IsInstance(obj, THPCharTensorClass)
#define THPByteTensor_Check(obj)    PyObject_IsInstance(obj, THPByteTensorClass)

#define THPDoubleTensor_CData(obj)  (obj)->cdata
#define THPFloatTensor_CData(obj)   (obj)->cdata
#define THPHalfTensor_CData(obj)    (obj)->cdata
#define THPLongTensor_CData(obj)    (obj)->cdata
#define THPIntTensor_CData(obj)     (obj)->cdata
#define THPShortTensor_CData(obj)   (obj)->cdata
#define THPCharTensor_CData(obj)    (obj)->cdata
#define THPByteTensor_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THPTensorType               TH_CONCAT_3(THP,Real,TensorType)
#define THPTensorBaseStr            TH_CONCAT_STRING_2(Real,TensorBase)
#define THPTensorStateless          TH_CONCAT_2(Real,TensorStateless)
#define THPTensorStatelessType      TH_CONCAT_2(Real,TensorStatelessType)
#define THPTensor_stateless_(NAME)  TH_CONCAT_4(THP,Real,Tensor_stateless_,NAME)
#endif

// Sparse Tensors
#define THSPTensor                   TH_CONCAT_3(THSP,Real,Tensor)
#define THSPTensorStr                TH_CONCAT_STRING_3(torch.Sparse,Real,Tensor)
#define THSPTensorClass              TH_CONCAT_3(THSP,Real,TensorClass)
#define THSPTensor_(NAME)            TH_CONCAT_4(THSP,Real,Tensor_,NAME)

#define THSPDoubleTensor_Check(obj)  PyObject_IsInstance(obj, THSPDoubleTensorClass)
#define THSPFloatTensor_Check(obj)   PyObject_IsInstance(obj, THSPFloatTensorClass)
#define THSPLongTensor_Check(obj)    PyObject_IsInstance(obj, THSPLongTensorClass)
#define THSPIntTensor_Check(obj)     PyObject_IsInstance(obj, THSPIntTensorClass)
#define THSPShortTensor_Check(obj)   PyObject_IsInstance(obj, THSPShortTensorClass)
#define THSPCharTensor_Check(obj)    PyObject_IsInstance(obj, THSPCharTensorClass)
#define THSPByteTensor_Check(obj)    PyObject_IsInstance(obj, THSPByteTensorClass)

#define THSPDoubleTensor_CData(obj)  (obj)->cdata
#define THSPFloatTensor_CData(obj)   (obj)->cdata
#define THSPLongTensor_CData(obj)    (obj)->cdata
#define THSPIntTensor_CData(obj)     (obj)->cdata
#define THSPShortTensor_CData(obj)   (obj)->cdata
#define THSPCharTensor_CData(obj)    (obj)->cdata
#define THSPByteTensor_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THSPTensorType               TH_CONCAT_3(THSP,Real,TensorType)
#define THSPTensorBaseStr            TH_CONCAT_STRING_3(Sparse,Real,TensorBase)
#define THSPTensorStateless          TH_CONCAT_3(Sparse,Real,TensorStateless)
#define THSPTensorStatelessType      TH_CONCAT_3(Sparse,Real,TensorStatelessType)
#define THSPTensor_stateless_(NAME)  TH_CONCAT_4(THSP,Real,Tensor_stateless_,NAME)
#endif

#include "generic/Tensor.h"
#include <TH/THGenerateAllTypes.h>

#include "generic/Tensor.h"
#include <TH/THGenerateHalfType.h>

#endif
