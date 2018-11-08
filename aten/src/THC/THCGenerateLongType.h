#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateLongType.h"
#endif

#define scalar_t int64_t
#define accreal int64_t
#define Real Long
#define CReal CudaLong
#define THC_REAL_IS_LONG
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_LONG

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
