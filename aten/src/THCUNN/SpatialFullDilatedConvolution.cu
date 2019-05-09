#include <THCUNN/THCUNN.h>
#include <THCUNN/im2col.h>
#include <THC/THCTensor.hpp>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <THCUNN/generic/SpatialFullDilatedConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
