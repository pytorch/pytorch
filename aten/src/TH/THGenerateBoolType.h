#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateBoolType.h"
#endif

#define scalar_t bool
#define ureal bool
#define accreal int64_t
#define Real Bool
#define TH_CONVERT_REAL_TO_ACCREAL(_val) (accreal)(_val)
#define TH_CONVERT_ACCREAL_TO_REAL(_val) (scalar_t)(_val)
#define TH_REAL_IS_BOOL
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef ureal
#undef accreal
#undef Real
#undef TH_REAL_IS_BOOL
#undef TH_CONVERT_REAL_TO_ACCREAL
#undef TH_CONVERT_ACCREAL_TO_REAL

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
