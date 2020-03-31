#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateComplexDoubleType.h"
#endif

#define scalar_t thust::complex<double>
#define accreal thust::complex<double>
#define Real ComplexDouble
#define CReal Cuda
#define THC_REAL_IS_COMPLEXDOUBLE
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_COMPLEXDOUBLE

#ifndef THCGenerateAllTypes
#ifndef THCGenerateFloatTypes
#undef THC_GENERIC_FILE
#endif
#endif
