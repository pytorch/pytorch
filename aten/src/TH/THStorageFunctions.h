#pragma once

#include <TH/THGeneral.h>

#define THStorage_(NAME) TH_CONCAT_4(TH,Real,Storage_,NAME)

#include <TH/generic/THStorage.h>
#include <TH/THGenerateByteType.h>

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
TH_API void THStorage_free(THStorage *storage);
