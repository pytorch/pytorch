#ifndef THIMG_H
#define THIMG_H

#include <stdbool.h>
#include <TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THIMG_(NAME) TH_CONCAT_3(THIMG_, Real, NAME)

#include "generic/THIMG.h"
#include <THGenerateFloatTypes.h>

#endif  // THIMG_H