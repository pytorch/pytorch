#include "THC.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCAsmUtils.cuh"
#include "THCScanUtils.cuh"
#include "THCTensorTypeUtils.cuh"
#include "THCTensorMathReduce.cuh"
#include <algorithm> // for std::min

#include "THCTensorKthValue.cuh"

#include "generic/THCTensorKthValue.cu"
#include "THCGenerateAllTypes.h"
