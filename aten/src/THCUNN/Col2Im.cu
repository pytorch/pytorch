#include <THCUNN/THCUNN.h>
#include <THCUNN/common.h>
#include <THCUNN/im2col.h>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>

#include <TH/THHalf.h>
#include <THCUNN/THCHalfAutoNumerics.cuh>

#include <THCUNN/generic/Col2Im.cu>
#include <THC/THCGenerateFloatTypes.h>
