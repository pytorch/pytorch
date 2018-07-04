#include "THCUNN.h"
#include "common.h"
#include "im2col.h"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCTensor.hpp"
#include "THCStorage.hpp"

#include "generic/Im2Col.cu"
#include "THCGenerateFloatTypes.h"
