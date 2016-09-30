#ifndef THCP_TENSOR_INC
#define THCP_TENSOR_INC

class THCPAutoGPU {
public:
  THCPAutoGPU(PyObject *args, PyObject *self=NULL);
  ~THCPAutoGPU();
  bool setDevice(PyObject *obj);
  int device = -1;
};

#define THCPTensor TH_CONCAT_3(THCP,Real,Tensor)
#define THCPTensorStr TH_CONCAT_STRING_3(torch.cuda.,Real,Tensor)
#define THCPTensorClass TH_CONCAT_3(THCP,Real,TensorClass)
#define THCPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)

#define THCPDoubleTensor_Check(obj)  PyObject_IsInstance(obj, THCPDoubleTensorClass)
#define THCPFloatTensor_Check(obj)   PyObject_IsInstance(obj, THCPFloatTensorClass)
#define THCPHalfTensor_Check(obj)   PyObject_IsInstance(obj, THCPHalfTensorClass)
#define THCPLongTensor_Check(obj)    PyObject_IsInstance(obj, THCPLongTensorClass)
#define THCPIntTensor_Check(obj)     PyObject_IsInstance(obj, THCPIntTensorClass)
#define THCPShortTensor_Check(obj)   PyObject_IsInstance(obj, THCPShortTensorClass)
#define THCPCharTensor_Check(obj)    PyObject_IsInstance(obj, THCPCharTensorClass)
#define THCPByteTensor_Check(obj)    PyObject_IsInstance(obj, THCPByteTensorClass)

#define THCPDoubleTensor_CData(obj)  (obj)->cdata
#define THCPFloatTensor_CData(obj)   (obj)->cdata
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

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Tensor.h"
#include <THC/THCGenerateAllTypes.h>

#endif
