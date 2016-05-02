#ifndef THP_STORAGE_INC
#define THP_STORAGE_INC

#define THPStorage_(NAME) TH_CONCAT_4(THP,Real,Storage_,NAME)
#define THPStorage TH_CONCAT_3(THP,Real,Storage)
#define THPStorageType TH_CONCAT_3(THP,Real,StorageType)
#define THPStorageBaseStr TH_CONCAT_STRING_2(Real,StorageBase)
#define THPStorageStr TH_CONCAT_STRING_2(Real,Storage)
#define THPStorageClass TH_CONCAT_3(THP,Real,StorageClass)

#include "generic/Storage.h"
#include <TH/THGenerateAllTypes.h>

#endif
