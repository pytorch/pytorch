#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateComplexTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THComplexLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include <TH/THGenerateComplexFloatType.h>
#include <TH/THGenerateComplexDoubleType.h>

#ifdef THComplexLocalGenerateManyTypes
#undef THComplexLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
