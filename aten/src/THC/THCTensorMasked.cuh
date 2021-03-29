#ifndef THC_TENSOR_MASKED_CUH
#define THC_TENSOR_MASKED_CUH
#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCReduce.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
#include <thrust/system/cuda/execution_policy.h>
#endif

template <typename T, typename MaskT, typename MaskPrefixSumT>
struct TensorMaskedSelectOp {
  TensorMaskedSelectOp(T* t) : out(t) {}
  __device__ inline void operator()(MaskT* mask,
                                    MaskPrefixSumT* maskPrefixSum,
                                    T* in) {
    if (*mask) {
      out[*maskPrefixSum] = *in;
    }
  }

  T* out;
};

#endif // THC_TENSOR_MASKED_CUH
