#ifndef THCS_GENERIC_FILE
#error "You must define THCS_GENERIC_FILE before including THGenerateHalfType.h"
#endif

#include <THC/THCHalf.h>

#ifdef CUDA_HALF_TENSOR

#define real half
#define accreal float
#define Real Half
#define CReal CudaHalf
#define THCS_REAL_IS_HALF
#line 1 THCS_GENERIC_FILE
#include THCS_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THCS_REAL_IS_HALF

#endif // CUDA_HALF_TENSOR

#ifndef THCSGenerateAllTypes
#ifndef THCSGenerateFloatTypes
#undef THCS_GENERIC_FILE
#endif
#endif
