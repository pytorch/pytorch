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
    *dst = ScalarConvert<T, T>::to(__ldg(src));
#else
    *dst = ScalarConvert<T, T>::to(*src);
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

template <typename TypeDst>
struct CopyOp<TypeDst, thrust::complex<float>> {
  __device__ __forceinline__ void operator()(TypeDst* dst, thrust::complex<float>* src) {
#if __CUDA_ARCH__ >= 350
    cuComplex temp = __ldg(reinterpret_cast<cuComplex*>(src));
    *dst = ScalarConvert<thrust::complex<float>, TypeDst>::to(thrust::complex<float>(temp.x, temp.y));
#else
    *dst = ScalarConvert<thrust::complex<float>, TypeDst>::to(*src);
#endif
  }
};

template <typename TypeDst>
struct CopyOp<TypeDst, thrust::complex<double>> {
  __device__ __forceinline__ void operator()(TypeDst* dst, thrust::complex<double>* src) {
#if __CUDA_ARCH__ >= 350
    cuDoubleComplex temp = __ldg(reinterpret_cast<cuDoubleComplex*>(src));
    *dst = ScalarConvert<thrust::complex<double>, TypeDst>::to(thrust::complex<double>(temp.x, temp.y));
#else
    *dst = ScalarConvert<thrust::complex<double>, TypeDst>::to(*src);
#endif
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
