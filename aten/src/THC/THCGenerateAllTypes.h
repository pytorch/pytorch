#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define THCGenerateAllTypes

#define THCTypeIdxByte   1
#define THCTypeIdxChar   2
#define THCTypeIdxShort  3
#define THCTypeIdxInt    4
#define THCTypeIdxLong   5
#define THCTypeIdxFloat  6
#define THCTypeIdxDouble 7
#define THCTypeIdxHalf   8
#define THCTypeIdx_(T) TH_CONCAT_2(THCTypeIdx,T)

#include "THCGenerateByteType.h"
#include "THCGenerateCharType.h"
#include "THCGenerateShortType.h"
#include "THCGenerateIntType.h"
#include "THCGenerateLongType.h"
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

#undef THCGenerateAllTypes
#undef THC_GENERIC_FILE
