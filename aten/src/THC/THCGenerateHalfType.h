#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include <TH/THHalf.h>

#define scalar_t THHalf
#define accreal float
#define Real Half

#define CReal CudaHalf

#define THC_REAL_IS_HALF
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real

#undef CReal

#undef THC_REAL_IS_HALF

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
