#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZMath.h"
#else

#include "THZTypeMacros.h"

static inline ntype THZMath_(sigmoid)(ntype value) {
    return 1.0 / (1.0 + THZ_MATH_NAME(exp)(-value));
}

static inline ntype THZMath_(rsqrt)(ntype x) {
    return 1.0 / THZ_MATH_NAME(sqrt)(x);
}

static inline ntype THZMath_(lerp)(ntype a, ntype b, ntype weight) {
    return a + weight * (b-a);
}

static inline ntype THZMath_(log1p)(ntype x) {
    return THZ_MATH_NAME(log)(1 + x);
}

#endif