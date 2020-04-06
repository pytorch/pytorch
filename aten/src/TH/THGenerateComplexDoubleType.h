#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateComplexDoubleType.h"
#endif

extern "C" {
  #define scalar_t double complex
}


#define accreal double
#define Real ComplexDouble
#define TH_REAL_IS_COMPLEXDOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef scalar_t
#undef Real
#undef TH_REAL_IS_COMPLEXDOUBLE

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
