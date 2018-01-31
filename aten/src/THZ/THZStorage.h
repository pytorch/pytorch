#ifndef THZ_STORAGE_INC
#define THZ_STORAGE_INC

#include "TH.h"

#define THZStorage        TH_CONCAT_3(TH,NType,Storage)
#define THZStorage_(NAME) TH_CONCAT_4(TH,NType,Storage_,NAME)

#include "generic/THZStorage.h"
#include "THZGenerateAllTypes.h"

#include "generic/THZStorageCopy.h"
#include "THZGenerateAllTypes.h"

#endif