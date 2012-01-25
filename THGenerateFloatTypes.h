#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateAllTypes.h"
#endif

#define real float
#define accreal double
#define Real Float
#define TH_REAL_IS_FLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef TH_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define TH_REAL_IS_DOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef TH_REAL_IS_DOUBLE

#undef TH_GENERIC_FILE
