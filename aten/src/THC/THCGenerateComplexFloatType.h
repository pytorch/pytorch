#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateComplexFloatType.h"
#endif

#define scalar_t c10::complex<float>
#define accreal c10::complex<float>
#define Real ComplexFloat

#define CReal CudaComplexFloat

#define THC_REAL_IS_COMPLEXFLOAT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real

#undef CReal

#undef THC_REAL_IS_COMPLEXFLOAT

#ifndef THCGenerateAllTypes
#ifndef THCGenerateComplexTypes
#undef THC_GENERIC_FILE
#endif
#endif
