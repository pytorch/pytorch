#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateComplexTypes.h"
#endif

#define THCGenerateComplexTypes

#define THCTypeIdxComplexFloat  1
#define THCTypeIdxComplexDouble 2

#define THCTypeIdx_(T) TH_CONCAT_2(THCTypeIdx,T)

#include <THC/THCGenerateComplexFloatType.h>
#include <THC/THCGenerateComplexDoubleType.h>

#undef THCTypeIdxComplexFloat
#undef THCTypeIdxComplexDouble
#undef THCTypeIdx_

#undef THCGenerateComplexTypes
#undef THC_GENERIC_FILE
