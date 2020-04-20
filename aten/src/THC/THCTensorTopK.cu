#include <THC/THC.h>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMath.h>
#include <THC/THCAsmUtils.cuh>
#include <THC/THCScanUtils.cuh>
#include <THC/THCTensorTypeUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <ATen/WrapDimUtils.h>
#include <algorithm> // for std::min

#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <THC/THCTensorTopK.cuh>

#include <THC/generic/THCTensorTopK.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorTopK.cu>
#include <THC/THCGenerateBFloat16Type.h>
