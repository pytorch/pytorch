#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>
#include <ATen/native/cuda/im2col.cuh>
#include <THC/THCTensor.hpp>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <THCUNN/generic/SpatialFullDilatedConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
