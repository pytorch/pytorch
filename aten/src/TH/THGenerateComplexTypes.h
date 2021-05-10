#ifndef TH_GENERIC_FILE
#error \
    "You must define TH_GENERIC_FILE before including THGenerateComplexTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THComplexLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include <TH/THGenerateComplexDoubleType.h>
#include <TH/THGenerateComplexFloatType.h>

#ifdef THComplexLocalGenerateManyTypes
#undef THComplexLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
