#ifndef THCP_STORAGE_INC
#define THCP_STORAGE_INC

#define THCPStorageStr TH_CONCAT_STRING_3(torch.cuda.,Real,Storage)
#define THCPStorageClass TH_CONCAT_3(THCP,Real,StorageClass)
#define THCPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)

#define THCPByteStorage_Check(obj) \
    PyObject_IsInstance(obj, THCPByteStorageClass)

#define THCPByteStorage_CData(obj)               (obj)->cdata

#define THCPStorageType TH_CONCAT_3(THCP,Real,StorageType)
#define THCPStorageBaseStr TH_CONCAT_STRING_3(Cuda,Real,StorageBase)

#include <torch/csrc/cuda/override_macros.h>

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.h"
#include <torch/csrc/THCGenerateByteType.h>

#endif
