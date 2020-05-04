#pragma once

#include <TH/THGeneral.h>
#include <TH/THAllocator.h>

#define THStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)

#include <TH/generic/THStorage.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THStorage.h>
#include <TH/THGenerateComplexTypes.h>

#include <TH/generic/THStorage.h>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THStorage.h>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THStorage.h>
#include <TH/THGenerateQTypes.h>

#include <TH/generic/THStorage.h>
#include <TH/THGenerateBFloat16Type.h>

#include <TH/generic/THStorageCopy.h>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THStorageCopy.h>
#include <TH/THGenerateComplexTypes.h>

#include <TH/generic/THStorageCopy.h>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THStorageCopy.h>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THStorageCopy.h>
#include <TH/THGenerateQTypes.h>

#include <TH/generic/THStorageCopy.h>
#include <TH/THGenerateBFloat16Type.h>

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
TH_API void THStorage_free(THStorage *storage);
