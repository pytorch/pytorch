// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif

#include <c10/util/MathConstants.h>

// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <math.h>

static_assert(M_PI == c10::pi<double>, "c10::pi<double> must be equal to M_PI");
static_assert(
    M_SQRT1_2 == c10::frac_sqrt_2<double>,
    "c10::frac_sqrt_2<double> must be equal to M_SQRT1_2");
