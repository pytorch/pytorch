#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"
#include "THCGeneral.h"

#define THCStorage        TH_CONCAT_3(TH,CReal,Storage)
#define THCStorage_(NAME) TH_CONCAT_4(TH,CReal,Storage_,NAME)

/* fast access methods */
#define THC_STORAGE_GET(storage, idx) ((storage)->data[(idx)])
#define THC_STORAGE_SET(storage, idx, value) ((storage)->data[(idx)] = (value))

#include "generic/THCStorage.h"
#include "THCGenerateAllTypes.h"

#endif
