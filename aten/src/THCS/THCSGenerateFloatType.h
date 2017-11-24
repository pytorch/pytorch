#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define real float
/* FIXME: fp64 has bad performance on some platforms; avoid using it unless
   we opt into it? */
#define accreal float
#define Real Float
#define CReal Cuda
#define THCS_REAL_IS_FLOAT
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_FLOAT

#ifndef THCSGenerateAllTypes
#ifndef THCSGenerateFloatTypes
#undef THCS_GENERIC_FILE
#endif
#endif
