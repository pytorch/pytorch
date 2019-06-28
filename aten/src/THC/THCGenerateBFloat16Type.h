#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THCGenerateBFloat16Type.h"
#endif
#include <c10/util/BFloat16.h>

#define scalar_t at::BFloat16
#define accreal float
#define Real BFloat16

#define CReal CudaBFloat16

#define THC_REAL_IS_BFLOAT16
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real

#undef CReal

#undef THC_REAL_IS_BFLOAT16

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
