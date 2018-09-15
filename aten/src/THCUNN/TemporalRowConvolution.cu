#include "THCUNN.h"
#include "common.h"
#include "row2col.h"

#include "TH/THHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCTensor.hpp"
#include "THCStorage.hpp"

#include "generic/TemporalRowConvolution.cu"

#include "THCGenerateFloatTypes.h"
