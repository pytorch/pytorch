#ifndef THC_TENSORMATH_POINTWISE_CUH
#define THC_TENSORMATH_POINTWISE_CUH

#include <type_traits>
#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCTensorCopy.h"
#include "THCApply.cuh"
#include "THCNumerics.cuh"
#include "THCReduce.cuh"


template <typename T>
struct TensorATan2Op {
  __device__ __forceinline__ void operator()(T* out, T* a, T* b) {
    *out = THCNumerics<T>::atan2(*a, *b);
  }
};

template <typename T>
struct TensorSigmoidOp {
  __device__ __forceinline__ void operator()(T* out, T* in) const {
    T one = (T) 1.0;
    *out = one / (one + THCNumerics<T>::exp(- *in));
  }

  __device__ __forceinline__ void operator()(T* v) const {
    T one = (T) 1.0;
    *v = one / (one + THCNumerics<T>::exp(- *v));
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSigmoidOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    half one = ScalarConvert<int, half>::to(1);
    *out = hdiv(one, __hadd(one, hexp(__hneg(*in))));
#else
    float fin = __half2float(*in);
    *out = __float2half(1.0f / (1.0f + expf(- fin)));
#endif
  }

  __device__ __forceinline__ void operator()(half* v) const {
#ifdef CUDA_HALF_INSTRUCTIONS
    half one = ScalarConvert<int, half>::to(1);
    *v = hdiv(one, __hadd(one, hexp(__hneg(*v))));
#else
    float fv = __half2float(*v);
    *v = __float2half(1.0f / (1.0f + expf(- fv)));
#endif
  }
};
#endif

template <typename T>
struct TensorSignOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    T orig = *in;
    *out = (orig > 0) - (orig < 0);
  }

  __device__ __forceinline__ void operator()(T* v) {
    T orig = *v;
    *v = (orig > 0) - (orig < 0);
  }
};

template <>
struct TensorSignOp<unsigned char> {
  __device__ __forceinline__ void operator()(unsigned char* out, unsigned char* in) {
    unsigned char orig = *in;
    *out = (orig == 0) ? 0 : 1;
  }

  __device__ __forceinline__ void operator()(unsigned char* v) {
    unsigned char orig = *v;
    *v = (orig == 0) ? 0 : 1;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSignOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    half zero = ScalarConvert<int, half>::to(0);
    half orig = *in;
    *out = __float2half((float) __hgt(orig, zero) - (float) __hlt(orig, zero));
#else
    float orig = __half2float(*in);
    *out = __float2half((orig > 0) - (orig < 0));
#endif
  }

  __device__ __forceinline__ void operator()(half* v) {
#ifdef CUDA_HALF_INSTRUCTIONS
    half zero = ScalarConvert<int, half>::to(0);
    half orig = *v;
    *v = __float2half((float) __hgt(orig, zero) -  (float) __hlt(orig, zero));
#else
    float orig = __half2float(*v);
    *v = __float2half((orig > 0) - (orig < 0));
#endif
  }
};
#endif

template <typename T>
struct TensorAddOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out += *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 + *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorAddOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*out, *in);
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout += fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in1, *in2);
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 + fin2;
    *out = __float2half(fout);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorCAddOp {
  TensorCAddOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out += val * *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 + val * *in2;
  }

  T val;
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCAddOp<half> {
  TensorCAddOp(half v) : val(v) {}

  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*out, __hmul(val, *in));
#else
    float fout = __half2float(*out);
    float fval = __half2float(val);
    float fin = __half2float(*in);

    fout += fval * fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hadd(*in1, __hmul(val, *in2));
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fval = __half2float(val);

    float fout = fin1 + fval * fin2;
    *out = __float2half(fout);
#endif
  }

  half val;
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorSubOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out -= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 - *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorSubOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*out, *in);
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout -= fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*in1, *in2);
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 - fin2;
    *out = __float2half(fout);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorMulOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out *= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 * *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorMulOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*out, *in);
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout *= fin;
    *out = __float2half(fout);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hmul(*in1, *in2);
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 * fin2;
    *out = __float2half(fout);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template<typename T, int StaticExp>
struct TensorPowOp {
  TensorPowOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    if (StaticExp == 1) {
      *out = *in;
    } else if (StaticExp == 2) {
      *out = THCNumerics<T>::mul(*in, *in);
    } else if (StaticExp == 3) {
      T square = THCNumerics<T>::mul(*in, *in);
      *out = THCNumerics<T>::mul(square, *in);
    } else {
      *out = THCNumerics<T>::pow(*in, val);
    }
  }

  __device__ __forceinline__ void operator()(T* v) {
    if (StaticExp == 1) {
      *v = *v;
    } else if (StaticExp == 2) {
      *v = THCNumerics<T>::mul(*v, *v);
    } else if (StaticExp == 3) {
      *v = THCNumerics<T>::mul(THCNumerics<T>::mul(*v, *v), *v);
    } else {
      *v = THCNumerics<T>::pow(*v, val);
    }
  }

  const T val;
};

template<typename T>
struct TensorPowOp<T, -1> {
  TensorPowOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::cinv(*in);
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = THCNumerics<T>::cinv(*v);
  }

  const T val;
};

template<typename T>
struct TensorPowOp<T, -2> {
  TensorPowOp(T v) : val(v) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    T square = THCNumerics<T>::mul(*in, *in);
    *out = THCNumerics<T>::cinv(square);
  }

  __device__ __forceinline__ void operator()(T* v) {
    T square = THCNumerics<T>::mul(*v, *v);
    *v = THCNumerics<T>::cinv(square);
  }

  const T val;
};

template<typename T>
struct TensorTPowOp {
  TensorTPowOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::pow(val, *in);
  }

  __device__ __forceinline__ void operator()(T* v) {
    *v = THCNumerics<T>::pow(val, *v);
  }

  const T val;
};

template <typename T>
struct TensorCPowOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::pow(*out, *in);
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = THCNumerics<T>::pow(*in1, *in2);
  }
};

template <>
struct TensorCPowOp<float> {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = powf(*in1, *in2);
  }
};


template <>
struct TensorCPowOp<double> {
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = pow(*out, *in);
  }

  __device__ __forceinline__ void operator()(double* out, double* in1, double* in2) {
    *out = pow(*in1, *in2);
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCPowOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
    // No fp16 pow function yet
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout = powf(fout, fin);
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
    // No fp16 pow function yet
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = powf(fin1, fin2);
    *out = __float2half(fout);
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorDivOp {
  __device__ __forceinline__ void
  operator()(T* out, T* in) {
    *out /= *in;
  }

  __device__ __forceinline__ void
  operator()(T* out, T* in1, T* in2) {
    *out = *in1 / *in2;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorDivOp<half> {
  __device__ __forceinline__ void
  operator()(half* out, half* in) {
    // No fp16 div instruction yet
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout /= fin;
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void
  operator()(half* out, half* in1, half* in2) {
    // No fp16 div instruction yet
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 / fin2;
    *out = __float2half(fout);
  }
};
#endif // CUDA_HALF_TENSOR

template<typename T>
static __device__ __forceinline__
typename std::enable_if<std::is_signed<T>::value, bool>::type
modulo_wrap(T a, T b) {
  return (a != 0) && (a < 0) != (b < 0);
}

template<typename T>
static __device__ __forceinline__
typename std::enable_if<std::is_unsigned<T>::value, bool>::type
modulo_wrap(T a, T b) {
  return false;
}

template <typename T>
struct TensorCRemainderOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    T val =  *out % *in;
    if (modulo_wrap(val, *in)) {
      val += *in;
    }
    *out = val;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    T val = *in1 % *in2;
    if (modulo_wrap(val, *in2)) {
      val += *in2;
    }
    *out = val;
  }
};

template <>
struct TensorCRemainderOp<float> {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in != 0.f ? *out - *in * floorf(*out / *in) : NAN;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in2 != 0.f ? *in1 - *in2 * floorf(*in1 / *in2) : NAN;
  }
};

template <>
struct TensorCRemainderOp<double> {
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = *in != 0. ? *out - *in * floor(*out / *in) : NAN;
  }

  __device__ __forceinline__ void operator()(double* out, double* in1, double* in2) {
    *out = *in2 != 0. ? *in1 - *in2 * floor(*in1 / *in2) : NAN;
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCRemainderOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*out, __hmul(*in, hfloor(__hdiv(*out, *in))));
#else
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    *out = fin != 0 ? __float2half(fout - fin * floorf(fout / fin)) : __float2half(NAN);
#endif
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
#ifdef CUDA_HALF_INSTRUCTIONS
    *out = __hsub(*in1, __hmul(*in2, hfloor(__hdiv(*in1, *in2))));
#else
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    *out = fin2 != 0 ? __float2half(fin1 - fin2 * floorf(fin1 / fin2)) : __float2half(NAN);
#endif
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorCFmodOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = *out % *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 % *in2;
  }
};

template <>
struct TensorCFmodOp<float> {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = fmodf(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = fmodf(*in1, *in2);
  }
};

template <>
struct TensorCFmodOp<double> {
  __device__ __forceinline__ void operator()(double* out, double* in) {
    *out = fmod(*out, *in);
  }

  __device__ __forceinline__ void operator()(double* out, double* in1, double* in2) {
    *out = fmod(*in1, *in2);
  }
};

#ifdef CUDA_HALF_TENSOR
template <>
struct TensorCFmodOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
    *out = __float2half(fmodf(__half2float(*out), __half2float(*in)));
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
    *out = __float2half(fmodf(__half2float(*in1), __half2float(*in2)));
  }
};
#endif // CUDA_HALF_TENSOR

template <typename T>
struct TensorClampOp {
  TensorClampOp(T min, T max) : minValue(min), maxValue(max) {}
  __device__ __forceinline__ void operator()(T* out, T* in) {
    T val = THCNumerics<T>::lt(*in, maxValue) ? *in : maxValue;
    *out = THCNumerics<T>::gt(minValue, val) ? minValue : val;
  }

  __device__ __forceinline__ void operator()(T* v) {
    T val = THCNumerics<T>::lt(*v, maxValue) ? *v : maxValue;
    *v = THCNumerics<T>::gt(minValue, val) ? minValue : val;
  }

  const T minValue;
  const T maxValue;
};

template <typename T>
struct TensorLerpOp {
  TensorLerpOp(T w) : w(w) {}

  __device__ __forceinline__ void operator()(T *out, T *a, T *b) {
    *out = THCNumerics<T>::add(
      *a,
      THCNumerics<T>::mul(
          w,
          THCNumerics<T>::sub(*b, *a)
        )
    );
  }

  const T w;
};

template <typename T>
struct TensorCrossOp {
  TensorCrossOp(int64_t sx, int64_t sy, int64_t so) : sx(sx), sy(sy), so(so) {}

  __device__ __forceinline__ void operator()(T* out, T* x, T*y) {
    T val0 = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[1 * sx], y[2 * sy]),
        THCNumerics<T>::mul(x[2 * sx], y[1 * sy])
    );

    T val1 = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[2 * sx], y[0 * sy]),
        THCNumerics<T>::mul(x[0 * sx], y[2 * sy])
    );

    T val2 = THCNumerics<T>::sub(
        THCNumerics<T>::mul(x[0 * sx], y[1 * sy]),
        THCNumerics<T>::mul(x[1 * sx], y[0 * sy])
    );

    out[0 * so] = val0;
    out[1 * so] = val1;
    out[2 * so] = val2;
  }

  const int64_t sx, sy, so;
};

template <typename T>
struct TensorMaxOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::gt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = THCNumerics<T>::gt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMinOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::lt(*out, *in) ? *out : *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = THCNumerics<T>::lt(*in1, *in2) ? *in1 : *in2;
  }
};

template <typename T>
struct TensorMaxValueOp {
  TensorMaxValueOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out) {
    *out = THCNumerics<T>::gt(*out, val) ? *out : val;
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::gt(*in, val) ? *in : val;
  }

  T val;
};

template <typename T>
struct TensorMinValueOp {
  TensorMinValueOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out) {
    *out = THCNumerics<T>::lt(*out, val) ? *out : val;
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::lt(*in, val) ? *in : val;
  }

  T val;
};

template <typename T>
struct TensorAddCMulOp {
  TensorAddCMulOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = THCNumerics<T>::add(
      *out,
      THCNumerics<T>::mul(
        val,
        THCNumerics<T>::mul(*in1, *in2)
      )
    );
  }

  T val;
};

template <typename T>
struct TensorAddCDivOp {
  TensorAddCDivOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = THCNumerics<T>::add(
      *out,
      THCNumerics<T>::mul(
        val,
        THCNumerics<T>::div(*in1, *in2)
      )
    );
  }

  T val;
};

template <typename T>
struct TensorLShiftOp {
  __device__ __forceinline__ void
  operator()(T* out, T* in) {
    *out <<= *in;
  }

  __device__ __forceinline__ void
  operator()(T* out, T* in1, T* in2) {
    *out = *in1 << *in2;
  }
};

template <>
struct TensorLShiftOp<float> {
  __device__ __forceinline__ void
  operator()(float* out, float* in) {
    *out *= powf(2.0f, *in);
  }

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out = *in1 * powf(2.0f, *in2);
  }
};

template <>
struct TensorLShiftOp<double> {
  __device__ __forceinline__ void
  operator()(double* out, double* in) {
    *out *= pow(2.0, *in);
  }

  __device__ __forceinline__ void
  operator()(double* out, double* in1, double* in2) {
    *out = *in1 * pow(2.0, *in2);
  }
};

template <typename T>
struct TensorRShiftOp {
  __device__ __forceinline__ void
  operator()(T* out, T* in) {
    *out >>= *in;
  }

  __device__ __forceinline__ void
  operator()(T* out, T* in1, T* in2) {
    *out = *in1 >> *in2;
  }
};


template <>
struct TensorRShiftOp<float> {
  __device__ __forceinline__ void
  operator()(float* out, float* in) {
    *out /= powf(2.0f, *in);
  }

  __device__ __forceinline__ void
  operator()(float* out, float* in1, float* in2) {
    *out = *in1 / powf(2.0f, *in2);
  }
};

template <>
struct TensorRShiftOp<double> {
  __device__ __forceinline__ void
  operator()(double* out, double* in) {
    *out /= pow(2.0, *in);
  }

  __device__ __forceinline__ void
  operator()(double* out, double* in1, double* in2) {
    *out = *in1 / pow(2.0, *in2);
  }
};

template <typename T>
struct TensorBitAndOp {
  __device__ __forceinline__ void
  operator()(T* out, T* in) {
    *out &= *in;
  }

  __device__ __forceinline__ void
  operator()(T* out, T* in1, T* in2) {
    *out = *in1 & *in2;
  }
};

template <typename T>
struct TensorBitOrOp {
  __device__ __forceinline__ void
  operator()(T* out, T* in) {
    *out |= *in;
  }

  __device__ __forceinline__ void
  operator()(T* out, T* in1, T* in2) {
    *out = *in1 | *in2;
  }
};

template <typename T>
struct TensorBitXorOp {
  __device__ __forceinline__ void
  operator()(T* out, T* in) {
    *out ^= *in;
  }

  __device__ __forceinline__ void
  operator()(T* out, T* in1, T* in2) {
    *out = *in1 ^ *in2;
  }
};

/*
 * The following function was converted to CUDA form from code that comes
 * with the following copyright notice. It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
template <typename real, typename accreal>
struct TensorDigammaOp {
  __device__ __forceinline__ void
  operator()(real* out, real* in) {
    using compute_type = typename std::conditional<std::is_same<real, half>::value, accreal, real>::type;
    static const double PI_f64 = 3.14159265358979323846;
    static const compute_type PSI_10 = 2.25175258906672110764;
    static const compute_type A[] = {
       8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
       7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
       3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
       8.33333333333333333333E-2,
    };

    auto x = scalar_cast<compute_type>(*in);
    if (x == 0) {
      *out = scalar_cast<real>(INFINITY);
      return;
    }

    bool x_is_integer = x == floor(x);
    compute_type result = 0;
    if (x < 0) {
      if (x_is_integer) {
        *out = scalar_cast<real>(INFINITY);
        return;
      }
      // Rounding errors in tan's input can really affect the output
      // for extreme values, so we always perform this computation in double.
      result = scalar_cast<compute_type>(
          - PI_f64 / tan(PI_f64 * scalar_cast<double>(x)));
      x = 1 - x;
    }

    while (x < 10) {
      result -= 1 / x;
      x += 1;
    }
    if (x == 10) {
      *out = scalar_cast<real>(result + PSI_10);
      return;
    }

    compute_type y = 0;
    if (x < 1.0e17) {
      compute_type z = 1.0 / (x * x);

      compute_type polevl_result = 0;
      for (int i = 0; i <= 6; i++) {
        polevl_result = polevl_result * z + A[i];
      }
      y = z * polevl_result;
    }

    *out = scalar_cast<real>(log(x) - (0.5 / x) - y + result);
    return;
  }
};

template <typename real, typename accreal>
struct TensorTrigammaOp {
  using compute_type = typename std::conditional<std::is_same<real, half>::value, accreal, real>::type;
  __device__ __forceinline__ void
  operator()(real* out, real* in) {
    const compute_type PI = 3.14159265358979323846;
    compute_type x = ScalarConvert<real, compute_type>::to(*in);
    compute_type sign = +1;
    compute_type result = 0;
    if (x < 0.5f) {
      sign = -1;
      compute_type sin_pi_x = THCNumerics<compute_type>::sin(PI * x);
      result -= (PI * PI) / (sin_pi_x * sin_pi_x);
      x = 1 - x;
    }
    for (int i = 0; i < 6; ++i) {
      result += 1 / (x * x);
      x += 1;
    }
    const compute_type ixx = 1 / (x*x);
    result += (1 + 1 / (2*x) + ixx * (1.f/6 - ixx * (1.f/30 - ixx * (1.f/42)))) / x;
    *out = ScalarConvert<compute_type, real>::to(sign * result);
  }
};

#endif // THC_TENSORMATH_POINTWISE_CUH
