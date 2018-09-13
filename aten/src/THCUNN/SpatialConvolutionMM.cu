#include "THCUNN.h"
#include "THCTensor.hpp"
#include "common.h"
#include "im2col.h"

#include "TH/THHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "generic/SpatialConvolutionMM.cu"
#include "THCGenerateFloatTypes.h"
