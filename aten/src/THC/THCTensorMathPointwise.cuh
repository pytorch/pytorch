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

template <>
struct TensorSigmoidOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) const {
    float fin = __half2float(*in);
    *out = __float2half(1.0f / (1.0f + expf(- fin)));
  }

  __device__ __forceinline__ void operator()(half* v) const {
    float fv = __half2float(*v);
    *v = __float2half(1.0f / (1.0f + expf(- fv)));
  }
};

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

template <>
struct TensorSignOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
    float orig = __half2float(*in);
    *out = __float2half((orig > 0) - (orig < 0));
  }

  __device__ __forceinline__ void operator()(half* v) {
    float orig = __half2float(*v);
    *v = __float2half((orig > 0) - (orig < 0));
  }
};

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

template <>
struct TensorCAddOp<half> {
  TensorCAddOp(half v) : val(v) {}

  __device__ __forceinline__ void operator()(half* out, half* in) {
    float fout = __half2float(*out);
    float fval = __half2float(val);
    float fin = __half2float(*in);

    fout += fval * fin;
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fval = __half2float(val);

    float fout = fin1 + fval * fin2;
    *out = __float2half(fout);
  }

  half val;
};

template <typename T>
struct TensorMulOp {
  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out *= *in;
  }

  __device__ __forceinline__ void operator()(T* out, T* in1, T* in2) {
    *out = *in1 * *in2;
  }
};

template <>
struct TensorMulOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    fout *= fin;
    *out = __float2half(fout);
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    float fout = fin1 * fin2;
    *out = __float2half(fout);
  }
};

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

template <>
struct TensorCRemainderOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
    float fout = __half2float(*out);
    float fin = __half2float(*in);
    *out = fin != 0 ? __float2half(fout - fin * floorf(fout / fin)) : __float2half(NAN);
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
    float fin1 = __half2float(*in1);
    float fin2 = __half2float(*in2);
    *out = fin2 != 0 ? __float2half(fin1 - fin2 * floorf(fin1 / fin2)) : __float2half(NAN);
  }
};

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

template <>
struct TensorCFmodOp<half> {
  __device__ __forceinline__ void operator()(half* out, half* in) {
    *out = __float2half(fmodf(__half2float(*out), __half2float(*in)));
  }

  __device__ __forceinline__ void operator()(half* out, half* in1, half* in2) {
    *out = __float2half(fmodf(__half2float(*in1), __half2float(*in2)));
  }
};

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
    *out = THCNumerics<T>::lt(*out, val) ? val : *out;  // this order propagates NaN
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::lt(*in, val) ? val : *in;  // this order propagates NaN
  }

  T val;
};

template <typename T>
struct TensorMinValueOp {
  TensorMinValueOp(T v) : val(v) {}

  __device__ __forceinline__ void operator()(T* out) {
    *out = THCNumerics<T>::gt(*out, val) ? val : *out;  // this order propagates NaN
  }

  __device__ __forceinline__ void operator()(T* out, T* in) {
    *out = THCNumerics<T>::gt(*in, val) ? val : *in;  // this order propagates NaN
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

template <typename real, typename accreal>
struct TensorI0Op {
  using compute_type = typename std::conditional<std::is_same<real, half>::value, accreal, real>::type;
  __device__ __forceinline__ void
  operator()(real* out, real* in) {
    static const compute_type I0_A[] = {
      -4.41534164647933937950E-18,
      3.33079451882223809783E-17,
      -2.43127984654795469359E-16,
      1.71539128555513303061E-15,
      -1.16853328779934516808E-14,
      7.67618549860493561688E-14,
      -4.85644678311192946090E-13,
      2.95505266312963983461E-12,
      -1.72682629144155570723E-11,
      9.67580903537323691224E-11,
      -5.18979560163526290666E-10,
      2.65982372468238665035E-9,
      -1.30002500998624804212E-8,
      6.04699502254191894932E-8,
      -2.67079385394061173391E-7,
      1.11738753912010371815E-6,
      -4.41673835845875056359E-6,
      1.64484480707288970893E-5,
      -5.75419501008210370398E-5,
      1.88502885095841655729E-4,
      -5.76375574538582365885E-4,
      1.63947561694133579842E-3,
      -4.32430999505057594430E-3,
      1.05464603945949983183E-2,
      -2.37374148058994688156E-2,
      4.93052842396707084878E-2,
      -9.49010970480476444210E-2,
      1.71620901522208775349E-1,
      -3.04682672343198398683E-1,
      6.76795274409476084995E-1
    };

    static const compute_type I0_B[] = {
      -7.23318048787475395456E-18,
      -4.83050448594418207126E-18,
      4.46562142029675999901E-17,
      3.46122286769746109310E-17,
      -2.82762398051658348494E-16,
      -3.42548561967721913462E-16,
      1.77256013305652638360E-15,
      3.81168066935262242075E-15,
      -9.55484669882830764870E-15,
      -4.15056934728722208663E-14,
      1.54008621752140982691E-14,
      3.85277838274214270114E-13,
      7.18012445138366623367E-13,
      -1.79417853150680611778E-12,
      -1.32158118404477131188E-11,
      -3.14991652796324136454E-11,
      1.18891471078464383424E-11,
      4.94060238822496958910E-10,
      3.39623202570838634515E-9,
      2.26666899049817806459E-8,
      2.04891858946906374183E-7,
      2.89137052083475648297E-6,
      6.88975834691682398426E-5,
      3.36911647825569408990E-3,
      8.04490411014108831608E-1
    };

    compute_type x = ScalarConvert<real, compute_type>::to(*in);
    if (x < 0) {
      x = -x;
    }
    if (x < 8.0) {
      compute_type y = x / 2.0 -2.0;
      compute_type b0 = I0_A[0];
      compute_type b1 = 0.0;
      compute_type b2 = 0.0;
      for (int j=1; j<30; j++){
        b2 = b1;
        b1 = b0;
        b0 =y * b1 - b2 + I0_A[j];
      }
      compute_type chbevl_result = (0.5 * (b0 - b2));
      *out = scalar_cast<real>(chbevl_result*exp(x));
      return;
    }
    compute_type y = 32.0 / x - 2.0;
    compute_type b0 = I0_B[0];
    compute_type b1 = 0.0;
    compute_type b2 = 0.0;
    for (int j=1; j<25; j++){
      b2 = b1;
      b1 = b0;
      b0 =y * b1 - b2 + I0_B[j];
    }
    compute_type chbevl_result = (0.5 * (b0 - b2));
    *out = scalar_cast<real>(chbevl_result * exp(x) / sqrt(x));
    return;
  }
};

template <typename real, typename accreal>
struct TensorI1Op {
  using compute_type = typename std::conditional<std::is_same<real, half>::value, accreal, real>::type;
  __device__ __forceinline__ void
  operator()(real* out, real* in) {
    static const compute_type I1_A[] = {
      2.77791411276104639959E-18,
      -2.11142121435816608115E-17,
      1.55363195773620046921E-16,
      -1.10559694773538630805E-15,
      7.60068429473540693410E-15,
      -5.04218550472791168711E-14,
      3.22379336594557470981E-13,
      -1.98397439776494371520E-12,
      1.17361862988909016308E-11,
      -6.66348972350202774223E-11,
      3.62559028155211703701E-10,
      -1.88724975172282928790E-9,
      9.38153738649577178388E-9,
      -4.44505912879632808065E-8,
      2.00329475355213526229E-7,
      -8.56872026469545474066E-7,
      3.47025130813767847674E-6,
      -1.32731636560394358279E-5,
      4.78156510755005422638E-5,
      -1.61760815825896745588E-4,
      5.12285956168575772895E-4,
      -1.51357245063125314899E-3,
      4.15642294431288815669E-3,
      -1.05640848946261981558E-2,
      2.47264490306265168283E-2,
      -5.29459812080949914269E-2,
      1.02643658689847095384E-1,
      -1.76416518357834055153E-1,
      2.52587186443633654823E-1
    };

    static const compute_type I1_B[] = {
      7.51729631084210481353E-18,
      4.41434832307170791151E-18,
      -4.65030536848935832153E-17,
      -3.20952592199342395980E-17,
      2.96262899764595013876E-16,
      3.30820231092092828324E-16,
      -1.88035477551078244854E-15,
      -3.81440307243700780478E-15,
      1.04202769841288027642E-14,
      4.27244001671195135429E-14,
      -2.10154184277266431302E-14,
      -4.08355111109219731823E-13,
      -7.19855177624590851209E-13,
      2.03562854414708950722E-12,
      1.41258074366137813316E-11,
      3.25260358301548823856E-11,
      -1.89749581235054123450E-11,
      -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,
      -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,
      -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,
      -9.76109749136146840777E-3,
      7.78576235018280120474E-1
    };

    compute_type x = ScalarConvert<real, compute_type>::to(*in);
    compute_type z = fabs(x);
    compute_type res = x;
    if (z < 8.0) {
      compute_type y = z / 2.0 - 2.0;
      compute_type b0 = I1_A[0];
      compute_type b1 = 0.0;
      compute_type b2 = 0.0;
      for (int j=1; j<29; j++){
        b2 = b1;
        b1 = b0;
        b0 =y * b1 - b2 + I1_A[j];
      }
      // chbevl_result = (0.5 * (b0 - b2));
      res= 0.5 * (b0 - b2) * z * exp(z);
    }
    else{
      compute_type y = 32.0 / z - 2.0;
      compute_type b0 = I1_B[0];
      compute_type b1 = 0.0;
      compute_type b2 = 0.0;
      for (int j=1; j<25; j++){
        b2 = b1;
        b1 = b0;
        b0 =y * b1 - b2 + I1_B[j];
      }
      // chbevl_result = (0.5 * (b0 - b2));
      res = 0.5 * (b0 - b2) * exp(z) / sqrt(z);
    }
    if (x <0.0){res = -res;}
    *out = scalar_cast<real>(res);
    return;
  }
};
#endif // THC_TENSORMATH_POINTWISE_CUH
