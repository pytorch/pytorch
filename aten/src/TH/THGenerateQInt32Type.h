#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateQInt32Type.h"
#endif

#define quantized_t c10::qint32
#define scalar_t int32_t
#define Real QInt32
#define RealUnderlying Int
#define THQUANTIZED
#define THQINT32
#define TH_REAL_IS_BYTE
#line 1 TH_GENERIC_FILE
#include TH_GENERIC_FILE
#undef scalar_t
#undef quantized_t
#undef Real
#undef RealUnderlying
#undef TH_REAL_IS_BYTE
#undef THQINT32
#undef THQUANTIZED

#ifndef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
