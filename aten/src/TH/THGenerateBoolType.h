#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateBoolType.h"
#endif

#define scalar_t bool
#define accreal int64_t
#define Real Bool
#define TH_REAL_IS_BOOL
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef TH_REAL_IS_BOOL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
