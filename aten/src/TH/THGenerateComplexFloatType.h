#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateComplexFloatType.h"
#endif

extern "C" {
  #define scalar_t float complex
}

#define accreal double
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
