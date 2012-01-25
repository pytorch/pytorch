#include "THCGeneral.h"
#include "THCTensor.h"

#define real float
#define Real Cuda
#define TH_GENERIC_FILE "generic/THTensor.c"

#include "generic/THTensor.c"

#undef TH_GENERIC_FILE
#undef real
#undef Real
