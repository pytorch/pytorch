#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateByteType.h"
#endif

#define scalar_t uint8_t
#define accreal int64_t
#define Real Byte
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef accreal
#undef Real
#undef TH_REAL_IS_BYTE

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
