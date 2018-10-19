#include "THCUNN.h"
#include "im2col.h"
#include "THCTensor.hpp"

#include "TH/THHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "generic/SpatialFullDilatedConvolution.cu"
#include "THCGenerateFloatTypes.h"
