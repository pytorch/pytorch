#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCTensor.hpp>


#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <cfloat>

template <typename T>
struct TensorFillOp {
  TensorFillOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* v) { *v = val; }

  const T val;
};

#include <THC/generic/THCTensorMath.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMath.cu>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensorMath.cu>
#include <THC/THCGenerateBFloat16Type.h>
