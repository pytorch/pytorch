#include "THCUNN.h"
#include "im2col.h"
#include "THCTensor.hpp"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "generic/SpatialFullDilatedConvolution.cu"
#include "THCGenerateFloatTypes.h"
