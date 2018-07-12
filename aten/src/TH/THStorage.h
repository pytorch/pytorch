#pragma once

#include "THGeneral.h"
#include "THAllocator.h"

#define THStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)

#include "generic/THStorage.h"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.h"
#include "THGenerateHalfType.h"

#include "generic/THStorageCopy.h"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.h"
#include "THGenerateHalfType.h"

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
TH_API void THStorage_free(THStorage *storage);
TH_API void THStorage_weakFree(THStorage *storage);

TH_API THDescBuff THLongStorage_sizeDesc(const THLongStorage *size);
TH_API THLongStorage *THLongStorage_newInferSize(THLongStorage *size, ptrdiff_t nElement);
