#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THFloatLocalGenerateManyTypes
#define THIntLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include <TH/THGenerateFloatType.h>
#include <TH/THGenerateDoubleType.h>
#include <TH/THGenerateLongType.h>

#ifdef THFloatLocalGenerateManyTypes
#undef THFloatLocalGenerateManyTypes
#undef THIntLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
