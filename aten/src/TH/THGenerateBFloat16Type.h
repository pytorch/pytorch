#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateBFloat16Type.h"
#endif

#include <TH/THBFloat16.h>
#define scalar_t THBFloat16
#define accreal float
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (scalar_t)(_val)
#define Real BFloat16
#define TH_REAL_IS_BFLOAT16
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef TH_REAL_IS_BFLOAT16
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
