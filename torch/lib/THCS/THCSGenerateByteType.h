#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define real uint8_t
#define accreal int64_t
#define Real Byte
#define CReal CudaByte
#define THCS_REAL_IS_BYTE
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_BYTE

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
