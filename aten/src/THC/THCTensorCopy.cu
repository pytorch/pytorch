#include <THC/THCApply.cuh>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorCopy.hpp>
#include <type_traits>

// Copy operator for the pointwise apply kernel
template <typename TypeDst, typename TypeSrc>
struct CopyOp {
  __device__ __forceinline__ void operator()(TypeDst* dst, TypeSrc* src) {
#if __CUDA_ARCH__ >= 350
    *dst = ScalarConvert<TypeSrc, TypeDst>::to(__ldg(src));
#else
    *dst = ScalarConvert<TypeSrc, TypeDst>::to(*src);
#endif
  }
};

template <>
struct CopyOp <bool, bool> {
  __device__ __forceinline__ void operator()(bool* dst, bool* src) {
      *dst = ScalarConvert<bool, bool>::to(*src);
  }
};

#include <THC/generic/THCTensorCopy.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorCopy.cu>
#include <THC/THCGenerateBoolType.h>
