#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateComplexDoubleType.h"
#endif

#define scalar_t c10::complex<double>
#define accreal c10::complex<double>
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
