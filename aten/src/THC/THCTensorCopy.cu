#include <THC/THCApply.cuh>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorCopy.hpp>
#include <type_traits>
#include <c10/util/BFloat16.h>

// Copy operator for the pointwise apply kernel
template <typename T>
struct CopyOp {
  __device__ __forceinline__ void operator()(T* dst, T* src) {
#if __CUDA_ARCH__ >= 350
    *dst = c10::static_cast_with_inter_type<T, T>::apply(*src);
#else
    *dst = c10::static_cast_with_inter_type<T, T>::apply(*src);
#endif
  }
};

template <>
struct CopyOp <bool> {
  __device__ __forceinline__ void operator()(bool* dst, bool* src) {
      *dst = ScalarConvert<bool, bool>::to(*src);
  }
};

template <>
struct CopyOp <at::BFloat16> {
  __device__ __forceinline__ void operator()(at::BFloat16* dst, at::BFloat16* src) {
      *dst = ScalarConvert<at::BFloat16, at::BFloat16>::to(*src);
  }
};

#include <THC/generic/THCTensorCopy.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorCopy.cu>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCTensorCopy.cu>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensorCopy.cu>
#include <THC/THCGenerateBFloat16Type.h>
