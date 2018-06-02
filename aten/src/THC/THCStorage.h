#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"
#include "THCGeneral.h"

#define THCStorage_(NAME) TH_CONCAT_4(TH,CReal,Storage_,NAME)

#include "generic/THCStorage.h"
#include "THCGenerateAllTypes.h"

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
THC_API void THCStorage_free(THCState *state, THCStorage *self);

#endif
