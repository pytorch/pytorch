#ifndef THS_GENERIC_FILE
#error "You must define THS_GENERIC_FILE before including THSGenerateAllTypes.h"
#endif

#define real float
#define accreal double
#define Real Float
#define THSInf FLT_MAX
#define THS_REAL_IS_FLOAT
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef THSInf
#undef THS_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define THSInf DBL_MAX
#define THS_REAL_IS_DOUBLE
#line 1 THS_GENERIC_FILE
#include THS_GENERIC_FILE
#undef accreal
#undef real
#undef Real
#undef THSInf
#undef THS_REAL_IS_DOUBLE

#undef THS_GENERIC_FILE
