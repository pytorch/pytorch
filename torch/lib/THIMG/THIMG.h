#ifndef THIMG_H
#define THIMG_H

#include <stdbool.h>
#include <TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THIMG_(NAME) TH_CONCAT_3(THIMG_, Real, NAME)

/*
#define THIndexTensor THLongTensor
#define THIndexTensor_(NAME) THLongTensor_ ## NAME

#define THIntegerTensor THIntTensor
#define THIntegerTensor_(NAME) THIntTensor_ ## NAME

typedef long THIndex_t;
typedef int THInteger_t;
*/

#include "generic/THIMG.h"
#include <THGenerateFloatTypes.h>

#endif  // THIMG_H