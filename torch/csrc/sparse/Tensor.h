#ifndef THSP_TENSOR_INC
#define THSP_TENSOR_INC

#define THSPTensor        TH_CONCAT_3(THSP,Real,Tensor)
#define THSPTensorStr     TH_CONCAT_STRING_3(torch.sparse.,Real,Tensor)
#define THSPTensorClass   TH_CONCAT_3(THSP,Real,TensorClass)
#define THSPTensor_(NAME) TH_CONCAT_4(THSP,Real,Tensor_,NAME)

#define THSPDoubleTensor_Check(obj) \
  PyObject_IsInstance(obj, THSPDoubleTensorClass)
#define THSPFloatTensor_Check(obj) \
  PyObject_IsInstance(obj, THSPFloatTensorClass)
#define THSPLongTensor_Check(obj) \
  PyObject_IsInstance(obj, THSPLongTensorClass)
#define THSPIntTensor_Check(obj) \
  PyObject_IsInstance(obj, THSPIntTensorClass)
#define THSPShortTensor_Check(obj) \
  PyObject_IsInstance(obj, THSPShortTensorClass)
#define THSPCharTensor_Check(obj) \
  PyObject_IsInstance(obj, THSPCharTensorClass)
#define THSPByteTensor_Check(obj) \
  PyObject_IsInstance(obj, THSPByteTensorClass)

#define THSPDoubleTensor_CData(obj)  (obj)->cdata
#define THSPFloatTensor_CData(obj)   (obj)->cdata
#define THSPLongTensor_CData(obj)    (obj)->cdata
#define THSPIntTensor_CData(obj)     (obj)->cdata
#define THSPShortTensor_CData(obj)   (obj)->cdata
#define THSPCharTensor_CData(obj)    (obj)->cdata
#define THSPByteTensor_CData(obj)    (obj)->cdata

#ifdef _THP_CORE
#define THSPTensorType TH_CONCAT_3(THSP,Real,TensorType)
#define THSPTensorBaseStr TH_CONCAT_STRING_3(Sparse,Real,TensorBase)
#define THSPTensorStateless          TH_CONCAT_3(Sparse,Real,TensorStateless)
#define THSPTensorStatelessType      TH_CONCAT_3(Sparse,Real,TensorStatelessType)
#define THSPTensor_stateless_(NAME)  TH_CONCAT_4(THSP,Real,Tensor_stateless_,NAME)
#endif

#include "torch/csrc/sparse/generic/Tensor.h"
#include <THS/THSGenerateAllTypes.h>

#endif
