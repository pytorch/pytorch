#include <THCUNN/THCUNN.h>
#include <THC/THCTensor.hpp>
#include <THCUNN/common.h>
#include <ATen/native/cuda/im2col.cuh>

#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>

#include <THCUNN/generic/SpatialConvolutionMM.cu>
#include <THC/THCGenerateFloatTypes.h>

#include <THCUNN/generic/SpatialConvolutionMM.cu>
#include <THC/THCGenerateBFloat16Type.h>
