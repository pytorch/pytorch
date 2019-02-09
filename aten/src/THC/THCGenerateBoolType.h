#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THCGenerateBoolType.h"
#endif

// TODO: define accreal tyoe once the correct value is known.
#define scalar_t bool
#define ureal uint8_t
#define Real Bool
#define CReal CudaBool
#define THC_REAL_IS_BOOL
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef scalar_t
#undef ureal
#undef Real
#undef CReal
#undef THC_REAL_IS_BOOL

#ifndef THCGenerateBoolType
#undef THC_GENERIC_FILE
#endif
