#include <torch/csrc/cuda/undef_macros.h>

#define THPStoragePtr THCPStoragePtr
#define THPTensorPtr THCPTensorPtr

#define THWTensor THCTensor
#define THWTensor_(NAME) THCTensor_(NAME)

#define THPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)
#define THPStorageBaseStr THCPStorageBaseStr
#define THPStorageStr THCPStorageStr
#define THPStorageClass THCPStorageClass
#define THPStorageType THCPStorageType

#define THPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THPTensor THCPTensor
#define THPTensorStr THCPTensorStr
#define THPTensorBaseStr THCPTensorBaseStr
#define THPTensorClass THCPTensorClass
#define THPTensorType THCPTensorType

#define THPTensorStatelessType THCPTensorStatelessType
#define THPTensorStateless THCPTensorStateless


#define THSPTensorPtr THCSPTensorPtr

#define THSPTensor_(NAME) TH_CONCAT_4(THCSP,Real,Tensor_,NAME)
#define THSPTensor_stateless_(NAME) TH_CONCAT_4(THCSP,Real,Tensor_stateless_,NAME)
#define THSPTensor THCSPTensor
#define THSPTensorStr THCSPTensorStr
#define THSPTensorBaseStr THCSPTensorBaseStr
#define THSPTensorClass THCSPTensorClass
#define THSPTensorType THCSPTensorType

#define THSPTensorStatelessType THCSPTensorStatelessType
#define THSPTensorStateless THCSPTensorStateless


#define TH_GENERIC_FILE THC_GENERIC_FILE
