#include <c10/util/MathConstants.h>

#include <math.h>

static_assert(M_PI == c10::pi<double>, "c10::pi<double> must be equal to M_PI");
