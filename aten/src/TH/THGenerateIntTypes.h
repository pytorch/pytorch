#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateIntTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THIntLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include <TH/THGenerateByteType.h>
#include <TH/THGenerateCharType.h>
#include <TH/THGenerateShortType.h>
#include <TH/THGenerateIntType.h>
#include <TH/THGenerateLongType.h>

#ifdef THIntLocalGenerateManyTypes
#undef THIntLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
