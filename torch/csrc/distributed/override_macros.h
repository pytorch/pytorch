#include "undef_macros.h"

#define THWStoragePtr THDStoragePtr
#define THPStoragePtr THDPStoragePtr
#define THWTensorPtr THDTensorPtr
#define THPTensorPtr THDPTensorPtr

#define THWStorage THDStorage
#define THWStorage_(NAME) THDStorage_(NAME)
#define THWTensor THDTensor
#define THWTensor_(NAME) THDTensor_(NAME)

#define THPStorage_(NAME) TH_CONCAT_4(THDP,Real,Storage_,NAME)
#define THPStorage THDPStorage
#define THPStorageBaseStr THDPStorageBaseStr
#define THPStorageStr THDPStorageStr
#define THPStorageClass THDPStorageClass
#define THPStorageType THDPStorageType

#define THPTensor_(NAME) TH_CONCAT_4(THDP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THDP,Real,Tensor_stateless_,NAME)
#define THPTensor THDPTensor
#define THPTensorStr THDPTensorStr
#define THPTensorBaseStr THDPTensorBaseStr
#define THPTensorClass THDPTensorClass
#define THPTensorType THDPTensorType

#define THPTensorStatelessType THDPTensorStatelessType
#define THPTensorStateless THDPTensorStateless

#define LIBRARY_STATE_NOARGS
#define LIBRARY_STATE
#define TH_GENERIC_FILE THD_GENERIC_FILE

#define THHostTensor TH_CONCAT_3(TH,Real,Tensor)
#define THHostTensor_(NAME) TH_CONCAT_4(TH,Real,Tensor_,NAME)
#define THHostStorage TH_CONCAT_3(TH,Real,Storage)
#define THHostStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)
