#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include <TH/THHalf.h>
#define scalar_t THHalf
#define accreal float
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (scalar_t)(_val)
#define Real Half
#define THInf TH_HALF_BITS_TO_LITERAL(TH_HALF_INF)
#define TH_REAL_IS_HALF
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef THInf
#undef TH_REAL_IS_HALF
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
