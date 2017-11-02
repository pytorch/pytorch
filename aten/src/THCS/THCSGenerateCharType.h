#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateCharType.h"
#endif

#define real int8_t
#define accreal int64_t
#define Real Char
#define CReal CudaChar
#define THCS_REAL_IS_CHAR
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_CHAR

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
