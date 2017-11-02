#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#define real double
#define accreal double
#define Real Double
#define CReal CudaDouble
#define THCS_REAL_IS_DOUBLE
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_DOUBLE

#ifndef THCSGenerateAllTypes
#ifndef THCSGenerateFloatTypes
#undef THCS_GENERIC_FILE
#endif
#endif
