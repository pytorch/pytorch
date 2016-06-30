#ifndef THCP_STORAGE_INC
#define THCP_STORAGE_INC

#define THCPStorage_(NAME) TH_CONCAT_4(THCP,Real,Storage_,NAME)
#define THCPStorage TH_CONCAT_3(THCP,Real,Storage)
#define THCPStorageType TH_CONCAT_3(THCP,Real,StorageType)
#define THCPStorageBaseStr TH_CONCAT_STRING_2(CReal,StorageBase)
#define THCPStorageStr TH_CONCAT_STRING_2(CReal,Storage)
#define THCPStorageClass TH_CONCAT_3(THCP,Real,StorageClass)

#include "override_macros.h"

#define THC_GENERIC_FILE "torch/csrc/generic/Storage.h"
#include <THC/THCGenerateAllTypes.h>

#endif
