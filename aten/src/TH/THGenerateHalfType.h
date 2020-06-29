#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include <TH/THHalf.h>
#define scalar_t THHalf
#define accreal float
#define Real Half
#define TH_REAL_IS_HALF
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef TH_REAL_IS_HALF

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
