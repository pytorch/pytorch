#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateShortType.h"
#endif

#define real int16_t
#define accreal int64_t
#define Real Short
#define CReal CudaShort
#define THCS_REAL_IS_SHORT
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_SHORT

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
