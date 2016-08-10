#undef THPTensor_
#undef THPTensor_stateless_
#undef THPTensor
#undef THPTensorStr
#undef THPTensorType
#undef THPTensorBaseStr
#undef THPTensorClass

#undef THPTensorStatelessType
#undef THPTensorStateless

#undef THPStorage_
#undef THPStorage
#undef THPStorageType
#undef THPStorageBaseStr
#undef THPStorageStr
#undef THPStorageClass

#undef THStorage
#undef THStorage_
#undef THTensor
#undef THTensor_

#undef THStoragePtr
#undef THPStoragePtr
#undef THTensorPtr
#undef THPTensorPtr

#undef THCStoragePtr
#undef THCTensorPtr
#undef THCPStoragePtr
#undef THCPTensorPtr

#define THCStoragePtr TH_CONCAT_3(THC,Real,StoragePtr)
#define THCTensorPtr  TH_CONCAT_3(THC,Real,TensorPtr)
#define THCPStoragePtr TH_CONCAT_3(THCP,Real,StoragePtr)
#define THCPTensorPtr  TH_CONCAT_3(THCP,Real,TensorPtr)

#define THStoragePtr THCStoragePtr
#define THPStoragePtr THCPStoragePtr
#define THTensorPtr THCTensorPtr
#define THPTensorPtr THCPTensorPtr

#define THStorage THCStorage
#define THStorage_(NAME) THCStorage_(NAME)
#define THTensor THCTensor
#define THTensor_(NAME) THCTensor_(NAME)

#define THPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)
#define THPStorage TH_CONCAT_3(THCP,Real,Storage)
#define THPStorageType TH_CONCAT_3(THCP,Real,StorageType)
#define THPStorageBaseStr TH_CONCAT_STRING_3(Cuda,Real,StorageBase)
#define THPStorageStr TH_CONCAT_STRING_3(Cuda,Real,Storage)
#define THPStorageClass TH_CONCAT_3(THCP,Real,StorageClass)

#define THPTensor_(NAME) TH_CONCAT_4(THCP,Real,Tensor_,NAME)
#define THPTensor_stateless_(NAME) TH_CONCAT_4(THCP,Real,Tensor_stateless_,NAME)
#define THPTensor TH_CONCAT_3(THCP,Real,Tensor)
#define THPTensorStr TH_CONCAT_STRING_3(Cuda,Real,Tensor)
#define THPTensorType TH_CONCAT_3(THCP,Real,TensorType)
#define THPTensorBaseStr TH_CONCAT_STRING_3(Cuda,Real,TensorBase)
#define THPTensorClass TH_CONCAT_3(THCP,Real,TensorClass)

#define THPTensorStatelessType THCPTensorStatelessType
#define THPTensorStateless THCPTensorStateless

#undef THPUtils_
#define THPUtils_(NAME) THCPUtils_(NAME)

#undef LIBRARY_STATE
#undef LIBRARY_STATE_NOARGS
#define LIBRARY_STATE_NOARGS state
#define LIBRARY_STATE state,
#define TH_GENERIC_FILE THC_GENERIC_FILE
