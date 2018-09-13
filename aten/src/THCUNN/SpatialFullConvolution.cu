#include "THCUNN.h"
#include "im2col.h"

#include "TH/THHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "generic/SpatialFullConvolution.cu"
#include "THCGenerateFloatTypes.h"
