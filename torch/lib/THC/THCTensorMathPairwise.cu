#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCTensorMathCompareT.cuh"

template <typename T>
struct TensorAddConstantOp {
  TensorAddConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in + val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v += val;
  }

  const T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorAddConstantOp<half> {
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorAddConstantOp(half v) : val(v) {}
#else
  TensorAddConstantOp(half v) : fval(THC_half2float(v)) {}
#endif

  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin + fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hadd(*v, val);
#else
    float fv = __half2float(*v);
    fv += fval;
    *v = __float2half(fv);
#endif
  }

#ifdef CUDA_HALF_INSTRUCTIONS
  const half val;
#else
  const float fval;
#endif
};
#endif // CUDA_HALF_TENSOR


template <typename T>
struct TensorSubConstantOp {
  TensorSubConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in - val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v -= val;
  }

  const T val;
};


#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSubConstantOp<half> {
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorSubConstantOp(half v): val(THC_float2half(-(THC_half2float(v)))) {}
#else
  TensorSubConstantOp(half v): fval(-(THC_half2float(v))) {}
#endif

  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin + fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hadd(*v, val);
#else
    float fv = __half2float(*v);
    fv += fval;
    *v = __float2half(fv);
#endif
  }

#ifdef CUDA_HALF_INSTRUCTIONS
  const half val;
#else
  const float fval;
#endif
};
#endif // CUDA_HALF_TENSOR


template <typename T>
struct TensorMulConstantOp {
  TensorMulConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v *= val;
  }

  const T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorMulConstantOp<half> {
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorMulConstantOp(half v) : val(v) {}
#else
  TensorMulConstantOp(half v) : fval(THC_half2float(v)) {}
#endif

  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin * fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hmul(*v, val);
#else
    float fv = __half2float(*v);
    fv *= fval;
    *v = __float2half(fv);
#endif
  }

#ifdef CUDA_HALF_INSTRUCTIONS
  const half val;
#else
  const float fval;
#endif
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorDivConstantOp {
  TensorDivConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in / val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v /= val;
  }

  const T val;
};

template <>
struct TensorDivConstantOp<float> {
  TensorDivConstantOp(float v) : val(1.f / v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v *= val;
  }

  const float val;
};

template <>
struct TensorDivConstantOp<double> {
  TensorDivConstantOp(double v) : val(1. / v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = *in * val;
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v *= val;
  }

  const double val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorDivConstantOp<half> {
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorDivConstantOp(half v) : val(ScalarInv<half>::to(v)) {}
#else
  TensorDivConstantOp(half v) : fval(1.f / THC_half2float(v)) {}
#endif
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in, val);
#else
    float fin = __half2float(*in);
    float fout = fin * fval;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hmul(*v, val);
#else
    float fv = __half2float(*v);
    fv *= fval;
    *v = __float2half(fv);
#endif
  }

#ifdef CUDA_HALF_INSTRUCTIONS
  const half val;
#else
  const float fval;
#endif
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorRemainderOp {
  TensorRemainderOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in % val;
    if ((*out * val) < 0){
      *out += val;
    }
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = *v % val;
    if ((*v * val) < 0){
      *v += val;
    }
  }

  const T val;
};

template <>
struct TensorRemainderOp<float> {
  TensorRemainderOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in - val * floorf(*in / val);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = *v - val * floorf(*v / val);
  }

  const float val;
};

template <>
struct TensorRemainderOp<double> {
  TensorRemainderOp(double v) : val(v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = *in - val * floor(*in / val);
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v = *v - val * floor(*v / val);
  }

  const double val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorRemainderOp<half> {
#ifdef CUDA_HALF_INSTRUCTIONS
  TensorRemainderOp(half v) : val(v) {}
#else
  TensorRemainderOp(half v): fval(THC_half2float(v)) {}
#endif

  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*in,  __hmul(val, hfloor(__hdiv(*in,  val))));
#else
    float fin = __half2float(*in);
    float fout = fin - fval * floorf(fin / fval);
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *v = __hsub(*v, __hmul(val, hfloor(__hdiv(*v, val))));
#else
    float fv = __half2float(*v);
    fv = fv - fval * floorf(fv / fval);
    *v = __float2half(fv);
#endif
  }

#ifdef CUDA_HALF_INSTRUCTIONS
  const half val;
#else
  const float fval;
#endif
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorFmodOp {
  TensorFmodOp(T v) : val((float)v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = (T) fmodf((float) *in, val);
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = (T) fmodf((float) *v, val);
  }

  const float val;
};

template <>
struct TensorFmodOp<double> {
  TensorFmodOp(double v) : val(v) {}
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = fmod(*in, val);
  }

  __device__ __forceinline__ void operator()(double* v) {
    *v = fmod(*v, val);
  }

  const double val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorFmodOp<half> {
  TensorFmodOp(half v): fval(THC_half2float(v)) {}

  __device__ __forceinline__ void operator()(half* out, half* in) {
    *out = __float2half(fmodf(__half2float(*in), fval));
  }

  __device__ __forceinline__ void operator()(half* v) {
    *v = __float2half(fmodf(__half2float(*v), fval));
  }

  const float fval;
};
#endif // CUDA_HALF_TENSOR

template <typename T, int Upper>
struct TensorTriOp {
  TensorTriOp(T *start_, long stride0_, long stride1_, long k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__ int mask(T *in) {
    ptrdiff_t n = in - start;
    long row, col;
    if (stride0 > stride1)
    {
      row = (long) (n / stride0);
      col = (long) ((n % stride0) / stride1);
    }
    else
    {
      row = (long) ((n % stride1) / stride0);
      col = (long) (n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = mask(in) ? *in : ScalarConvert<int, T>::to(0);
  }

  __device__ __forceinline__ void operator()(T* v) {
    if (!mask(v))
      *v = ScalarConvert<int, T>::to(0);
  }

  const T *start;
  const long stride0, stride1, k;
};

template <typename T>
struct TensorLShiftConstantOp {
  TensorLShiftConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in << val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v <<= val;
  }

  const T val;
};

template <typename T>
struct TensorRShiftConstantOp {
  TensorRShiftConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in >> val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v >>= val;
  }

  const T val;
};

template <typename T>
struct TensorBitAndConstantOp {
  TensorBitAndConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in & val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v &= val;
  }

  const T val;
};

template <typename T>
struct TensorBitOrConstantOp {
  TensorBitOrConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in | val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v |= val;
  }

  const T val;
};

template <typename T>
struct TensorBitXorConstantOp {
  TensorBitXorConstantOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *in ^ val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v ^= val;
  }

  const T val;
};

#include "generic/THCTensorMathPairwise.cu"
#include "THCGenerateAllTypes.h"
