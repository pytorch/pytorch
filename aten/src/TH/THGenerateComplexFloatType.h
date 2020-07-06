#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateComplexFloatType.h"
#endif

#define scalar_t c10::complex<float>
#define accreal c10::complex<double>
#define Real ComplexFloat
#define TH_REAL_IS_COMPLEXFLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef scalar_t
#undef Real
#undef TH_REAL_IS_COMPLEXFLOAT

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
