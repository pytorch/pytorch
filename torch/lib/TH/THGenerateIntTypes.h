#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateIntTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THIntLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include "THGenerateByteType.h"
#include "THGenerateCharType.h"
#include "THGenerateShortType.h"
#include "THGenerateIntType.h"
#include "THGenerateLongType.h"

#ifdef THIntLocalGenerateManyTypes
#undef THIntLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
