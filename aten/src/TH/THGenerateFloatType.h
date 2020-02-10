#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatType.h"
#endif

#define scalar_t float
#define accreal double
#define Real Float
#define TH_REAL_IS_FLOAT
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef accreal
#undef scalar_t
#undef Real
#undef TH_REAL_IS_FLOAT

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
