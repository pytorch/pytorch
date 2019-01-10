#include "THCUNN.h"
#include "common.h"
#include "row2col.h"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCTensor.hpp"
#include "THCStorage.hpp"

#include "generic/TemporalRowConvolution.cu"

#include "THCGenerateFloatTypes.h"
