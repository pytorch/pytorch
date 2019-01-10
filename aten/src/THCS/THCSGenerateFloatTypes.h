#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#define THCSGenerateFloatTypes

#define THCSTypeIdxByte   1
#define THCSTypeIdxChar   2
#define THCSTypeIdxShort  3
#define THCSTypeIdxInt    4
#define THCSTypeIdxLong   5
#define THCSTypeIdxFloat  6
#define THCSTypeIdxDouble 7
#define THCSTypeIdxHalf   8
#define THCSTypeIdx_(T) TH_CONCAT_2(THCSTypeIdx,T)

#include "THCSGenerateHalfType.h"
#include "THCSGenerateFloatType.h"
#include "THCSGenerateDoubleType.h"

#undef THCSTypeIdxByte
#undef THCSTypeIdxChar
#undef THCSTypeIdxShort
#undef THCSTypeIdxInt
#undef THCSTypeIdxLong
#undef THCSTypeIdxFloat
#undef THCSTypeIdxDouble
#undef THCSTypeIdxHalf
#undef THCSTypeIdx_

#undef THCSGenerateFloatTypes
#undef THCS_GENERIC_FILE
