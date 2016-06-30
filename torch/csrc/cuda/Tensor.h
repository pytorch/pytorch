#ifndef THCP_TENSOR_INC
#define THCP_TENSOR_INC

#define THCPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)
#define THCPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THCPTensor TH_CONCAT_3(THCP,Real,Tensor)
#define THCPTensorStr TH_CONCAT_STRING_2(CReal,Tensor)
#define THCPTensorType TH_CONCAT_3(THCP,Real,TensorType)
#define THCPTensorBaseStr TH_CONCAT_STRING_2(CReal,TensorBase)
#define THCPTensorClass TH_CONCAT_3(THCP,Real,TensorClass)

#define THCPTensorStatelessType TH_CONCAT_2(CReal,TensorStatelessType)
#define THCPTensorStatelessMethods TH_CONCAT_2(CReal,TensorStatelessMethods)
#define THCPTensorStateless TH_CONCAT_2(CReal,TensorStateless)

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Tensor.h"
#include <THC/THCGenerateAllTypes.h>

#endif
