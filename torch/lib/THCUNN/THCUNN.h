#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THCIndexTensor THCudaLongTensor
#define THCIndexTensor_(NAME) THCudaLongTensor_ ## NAME
typedef long THCIndex_t;

#define THNN_(NAME) TH_CONCAT_3(THNN_, CReal, NAME)

#include "generic/THCUNN.h"
#include "THCGenerateFloatTypes.h"
