#include "general.h"

#define torch_TensorOperator_(NAME) TH_CONCAT_4(torch_,Real,TensorOperator_,NAME)
#define torch_Tensor_id TH_CONCAT_3(torch_,Real,Tensor_id)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#include "generic/TensorOperator.c"
#include "THGenerateAllTypes.h"
