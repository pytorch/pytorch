#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THCUNN/row2col.h>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>

#include <THCUNN/generic/TemporalRowConvolution.cu>

#include <THC/THCGenerateFloatTypes.h>
