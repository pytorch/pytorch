#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateIntType.h"
#endif

#define scalar_t int32_t
#define accreal int64_t
#define Real Int
#define TH_REAL_IS_INT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef TH_REAL_IS_INT

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
