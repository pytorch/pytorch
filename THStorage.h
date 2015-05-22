#ifndef TH_STORAGE_INC
#define TH_STORAGE_INC

#include "THGeneral.h"
#include "THAllocator.h"

#define THStorage        TH_CONCAT_3(TH,Real,Storage)
#define THStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)

/* fast access methods */
#define TH_STORAGE_GET(storage, idx) ((storage)->data[(idx)])
#define TH_STORAGE_SET(storage, idx, value) ((storage)->data[(idx)] = (value))

#define TH_Type() TH_CONCAT_2(TH_,Real)

#include "generic/THStorage.h"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.h"
#include "THGenerateAllTypes.h"

#endif
