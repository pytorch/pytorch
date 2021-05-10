#ifndef THC_GENERIC_FILE
#error \
    "You must define THC_GENERIC_FILE before including THCGenerateComplexTypes.h"
#endif

#define THCGenerateComplexTypes

#include <THC/THCGenerateComplexDoubleType.h>
#include <THC/THCGenerateComplexFloatType.h>

#undef THCGenerateComplexTypes
#undef THC_GENERIC_FILE
