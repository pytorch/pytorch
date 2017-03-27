#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define THGenerateAllTypes

#include "THGenerateFloatTypes.h"
#include "THGenerateIntTypes.h"

#undef THGenerateAllTypes
#undef TH_GENERIC_FILE
