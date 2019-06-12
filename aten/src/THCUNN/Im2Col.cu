#include "THCUNN.h"
#include "common.h"
#include "im2col.h"

#include "TH/THHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCTensor.hpp"
#include "THCStorage.hpp"

#include "generic/Im2Col.cu"
#include "THCGenerateFloatTypes.h"
