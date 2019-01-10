#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateIntType.h"
#endif

#define real int32_t
#define accreal int64_t
#define Real Int
#define CReal CudaInt
#define THCS_REAL_IS_INT
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_INT

#ifndef THCSGenerateAllTypes
#undef THCS_GENERIC_FILE
#endif
