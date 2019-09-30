#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include <TH/THStorageFunctions.h>
#include <THC/THCGeneral.h>

#define THCStorage_(NAME) TH_CONCAT_4(TH,CReal,Storage_,NAME)

#include <THC/generic/THCStorage.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorage.h>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorage.h>
#include <THC/THCGenerateBFloat16Type.h>

#endif
