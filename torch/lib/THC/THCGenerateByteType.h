#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define real unsigned char
#define accreal long
#define Real Byte
#define CReal CudaByte
#define THC_REAL_IS_BYTE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_BYTE

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
