#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateLongType.h"
#endif

#define real int64_t
#define accreal int64_t
#define Real Long
#define CReal CudaLong
#define THCS_REAL_IS_LONG
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_LONG

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
