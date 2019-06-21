#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>
#include <ATen/native/cuda/vol2col.cuh>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>

#include <THCUNN/generic/VolumetricDilatedConvolution.cu>
#include <THC/THCGenerateFloatTypes.h>
