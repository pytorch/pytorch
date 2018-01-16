#include "THC.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCAsmUtils.cuh"
#include "THCScanUtils.cuh"
#include "THCTensorTypeUtils.cuh"
#include "THCTensorMathReduce.cuh"
#include <algorithm> // for std::min

#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#include "THCTensorTopK.cuh"

#include "generic/THCTensorTopK.cu"
#include "THCGenerateAllTypes.h"

