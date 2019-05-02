#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include "THHalf.h"

#define scalar_t at::Half
#define accreal double
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (scalar_t)(_val)
#define Real Half
#define THInf std::mumeric_limits<at::Half>::max()
#define TH_REAL_IS_HALF
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef scalar_t
#undef Real
#undef THInf
#undef TH_REAL_IS_HALF
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
