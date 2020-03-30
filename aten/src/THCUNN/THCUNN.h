#include <THC/THC.h>

#define THCIndexTensor THCudaLongTensor
#define THCIndexTensor_(NAME) THCudaLongTensor_ ## NAME
typedef int64_t THCIndex_t;

#define THNN_(NAME) TH_CONCAT_3(THNN_, CReal, NAME)

#include <THCUNN/generic/THCUNN.h>
#include <THC/THCGenerateFloatTypes.h>

#include <THCUNN/generic/THCUNN.h>
#include <THC/THCGenerateBFloat16Type.h>
