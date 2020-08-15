#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateCharType.h"
#endif

#define scalar_t int8_t
#define accreal int64_t
#define Real Char
#define TH_REAL_IS_CHAR
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef TH_REAL_IS_CHAR

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
