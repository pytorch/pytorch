#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <ATen/native/cuda/im2col.cuh>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>

#include <THCUNN/generic/SpatialDilatedConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
