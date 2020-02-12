#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateQInt8Type.h"
#endif

#define quantized_t c10::qint8
#define scalar_t int8_t
#define Real QInt8
#define RealUnderlying Char
#define THQUANTIZED
#define THQINT8
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef quantized_t
#undef Real
#undef RealUnderlying
#undef TH_REAL_IS_BYTE
#undef THQINT8
#undef THQUANTIZED

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
