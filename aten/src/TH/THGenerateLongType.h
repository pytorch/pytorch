#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateLongType.h"
#endif

#define scalar_t int64_t
#define accreal int64_t
#define Real Long
#define TH_REAL_IS_LONG
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef TH_REAL_IS_LONG

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
