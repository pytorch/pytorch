#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>
#include <THCUNN/vol2col.h>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <THCUNN/generic/VolumetricFullDilatedConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
