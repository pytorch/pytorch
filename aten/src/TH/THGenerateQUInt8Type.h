#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateQUInt8Type.h"
#endif

#define quantized_t c10::quint8
#define scalar_t uint8_t
#define Real QUInt8
#define RealUnderlying Byte
#define THQUANTIZED
#define THQUINT8
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef quantized_t
#undef Real
#undef RealUnderlying
#undef TH_REAL_IS_BYTE
#undef THQUINT8
#undef THQUANTIZED

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
