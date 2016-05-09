#ifndef THP_TENSOR_INC
#define THP_TENSOR_INC

#define THPTensor_(NAME) TH_CONCAT_4(THP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THP,Real,Tensor_stateless_,NAME)
#define THPTensor TH_CONCAT_3(THP,Real,Tensor)
#define THPTensorType TH_CONCAT_3(THP,Real,TensorType)
#define THPTensorBaseStr TH_CONCAT_STRING_2(Real,TensorBase)
#define THPTensorClass TH_CONCAT_3(THP,Real,TensorClass)

#define THPTensorStatelessType TH_CONCAT_2(Real,TensorStatelessType)
#define THPTensorStatelessMethods TH_CONCAT_2(Real,TensorStatelessMethods)
#define THPTensorStateless TH_CONCAT_2(Real,TensorStateless)

#include "generic/Tensor.h"
#include <TH/THGenerateAllTypes.h>

#endif
