#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorageFunctions.h"
#include "THCGeneral.h"

#define THCStorage_(NAME) TH_CONCAT_4(TH,CReal,Storage_,NAME)

#include "generic/THCStorage.h"
#include "THCGenerateAllTypes.h"

#endif
