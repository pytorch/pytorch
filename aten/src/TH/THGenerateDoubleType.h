#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateDoubleType.h"
#endif

#define scalar_t double
#define accreal double
#define Real Double
#define TH_REAL_IS_DOUBLE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef scalar_t
#undef Real
#undef TH_REAL_IS_DOUBLE

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
