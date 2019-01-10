#include "THCUNN.h"
#include "im2col.h"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "generic/SpatialFullConvolution.cu"
#include "THCGenerateFloatTypes.h"
