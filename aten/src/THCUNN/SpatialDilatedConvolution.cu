#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THCUNN/im2col.h>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>

#include <THCUNN/generic/SpatialDilatedConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
