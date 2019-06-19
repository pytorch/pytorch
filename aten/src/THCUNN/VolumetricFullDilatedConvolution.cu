#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>
#include <ATen/native/cuda/vol2col.cuh>
#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <THCUNN/generic/VolumetricFullDilatedConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
