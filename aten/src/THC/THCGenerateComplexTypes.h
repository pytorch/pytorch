#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THCGenerateComplexTypes.h"
#endif

#define THCGenerateComplexTypes

#include <THC/THCGenerateComplexFloatType.h>
#include <THC/THCGenerateComplexDoubleType.h>

#undef THCGenerateComplexTypes
#undef THC_GENERIC_FILE
