#include "THCUNN.h"
#include "THCTensor.hpp"
#include "common.h"
#include "vol2col.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "generic/VolumetricDilatedConvolution.cu"
#include "THCGenerateFloatTypes.h"
