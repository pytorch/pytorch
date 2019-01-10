#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateCharType.h"
#endif

#define scalar_t int8_t
#define accreal int64_t
#define Real Char
#define CReal CudaChar
#define THC_REAL_IS_CHAR
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_CHAR

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
