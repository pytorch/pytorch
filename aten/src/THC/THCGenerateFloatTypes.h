#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#define THCGenerateFloatTypes

#define THCTypeIdxByte   1
#define THCTypeIdxChar   2
#define THCTypeIdxShort  3
#define THCTypeIdxInt    4
#define THCTypeIdxLong   5
#define THCTypeIdxFloat  6
#define THCTypeIdxDouble 7
#define THCTypeIdxHalf   8
#define THCTypeIdx_(T) TH_CONCAT_2(THCTypeIdx,T)

#include "THCGenerateHalfType.h"
#include "THCGenerateFloatType.h"
#include "THCGenerateDoubleType.h"

#undef THCTypeIdxByte
#undef THCTypeIdxChar
#undef THCTypeIdxShort
#undef THCTypeIdxInt
#undef THCTypeIdxLong
#undef THCTypeIdxFloat
#undef THCTypeIdxDouble
#undef THCTypeIdxHalf
#undef THCTypeIdx_

#undef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
