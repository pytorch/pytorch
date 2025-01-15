#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/macros/Macros.h>
#include <cmath>
#include <limits>

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
}

namespace at::native {

using structured_beta_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(structured_beta_fn, betainc_stub)

/* below codes are from 'third_party/eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h' */
// hard coded from eigen/Eigen/src/Core/util/Macros.h
#define EIGEN_STRONG_INLINE inline  // hard coded Macro
#define EIGEN_DEVICE_FUNC C10_HOST_DEVICE // hard coded Macro

template <typename Scalar>
struct cephes_helper {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar machep() { assert(false && "machep not supported for this type"); return 0.0; }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar big() { assert(false && "big not supported for this type"); return 0.0; }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar biginv() { assert(false && "biginv not supported for this type"); return 0.0; }
};

template <>
struct cephes_helper<float> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float machep() {
    return std::numeric_limits<float>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float big() {
    // use epsneg (1.0 - epsneg == 1.0)
    return 1.0f / (std::numeric_limits<float>::epsilon() / 2);
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float biginv() {
    // epsneg
    return machep();
  }
};

template <>
struct cephes_helper<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double machep() {
    return std::numeric_limits<double>::epsilon() / 2;  // 1.0 - machep == 1.0
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double big() {
    return 1.0 / std::numeric_limits<double>::epsilon();
  }
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double biginv() {
    // inverse of eps
    return std::numeric_limits<double>::epsilon();
  }
};

template <typename Scalar>
struct betainc_impl {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(Scalar, Scalar, Scalar) {
    /*    betaincf.c
     *
     *    Incomplete beta integral
     *
     *
     * SYNOPSIS:
     *
     * float a, b, x, y, betaincf();
     *
     * y = betaincf( a, b, x );
     *
     *
     * DESCRIPTION:
     *
     * Returns incomplete beta integral of the arguments, evaluated
     * from zero to x.  The function is defined as
     *
     *                  x
     *     -            -
     *    | (a+b)      | |  a-1     b-1
     *  -----------    |   t   (1-t)   dt.
     *   -     -     | |
     *  | (a) | (b)   -
     *                 0
     *
     * The domain of definition is 0 <= x <= 1.  In this
     * implementation a and b are restricted to positive values.
     * The integral from x to 1 may be obtained by the symmetry
     * relation
     *
     *    1 - betainc( a, b, x )  =  betainc( b, a, 1-x ).
     *
     * The integral is evaluated by a continued fraction expansion.
     * If a < 1, the function calls itself recursively after a
     * transformation to increase a to a+1.
     *
     * ACCURACY (float):
     *
     * Tested at random points (a,b,x) with a and b in the indicated
     * interval and x between 0 and 1.
     *
     * arithmetic   domain     # trials      peak         rms
     * Relative error:
     *    IEEE       0,30       10000       3.7e-5      5.1e-6
     *    IEEE       0,100      10000       1.7e-4      2.5e-5
     * The useful domain for relative error is limited by underflow
     * of the single precision exponential function.
     * Absolute error:
     *    IEEE       0,30      100000       2.2e-5      9.6e-7
     *    IEEE       0,100      10000       6.5e-5      3.7e-6
     *
     * Larger errors may occur for extreme ratios of a and b.
     *
     * ACCURACY (double):
     * arithmetic   domain     # trials      peak         rms
     *    IEEE      0,5         10000       6.9e-15     4.5e-16
     *    IEEE      0,85       250000       2.2e-13     1.7e-14
     *    IEEE      0,1000      30000       5.3e-12     6.3e-13
     *    IEEE      0,10000    250000       9.3e-11     7.1e-12
     *    IEEE      0,100000    10000       8.7e-10     4.8e-11
     * Outputs smaller than the IEEE gradual underflow threshold
     * were excluded from these statistics.
     *
     * ERROR MESSAGES:
     *   message         condition      value returned
     * incbet domain      x<0, x>1          nan
     * incbet underflow                     nan
     */


    TORCH_INTERNAL_ASSERT((std::is_same<Scalar, Scalar>::value == false),
                        "THIS_TYPE_IS_NOT_SUPPORTED");
    return Scalar(0.);
  }
};

/* Continued fraction expansion #1 for incomplete beta integral (small_branch = True)
 * Continued fraction expansion #2 for incomplete beta integral (small_branch = False)
 */
template <typename Scalar>
struct incbeta_cfe {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE Scalar run(Scalar a, Scalar b, Scalar x, bool small_branch) {
    TORCH_INTERNAL_ASSERT((std::is_same<Scalar, float>::value ||
                         std::is_same<Scalar, double>::value),
                        "THIS_TYPE_IS_NOT_SUPPORTED");
    const Scalar big = cephes_helper<Scalar>::big();
    const Scalar machep = cephes_helper<Scalar>::machep();
    const Scalar biginv = cephes_helper<Scalar>::biginv();

    const Scalar zero = 0;
    const Scalar one = 1;
    const Scalar two = 2;

    Scalar xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
    Scalar k1, k2, k3, k4, k5, k6, k7, k8, k26update;
    Scalar ans;
    int n;

    const int num_iters = (std::is_same<Scalar, float>::value) ? 100 : 300;
    const Scalar thresh =
        (std::is_same<Scalar, float>::value) ? machep : Scalar(3) * machep;
    Scalar r = (std::is_same<Scalar, float>::value) ? zero : one;

    if (small_branch) {
      k1 = a;
      k2 = a + b;
      k3 = a;
      k4 = a + one;
      k5 = one;
      k6 = b - one;
      k7 = k4;
      k8 = a + two;
      k26update = one;
    } else {
      k1 = a;
      k2 = b - one;
      k3 = a;
      k4 = a + one;
      k5 = one;
      k6 = a + b;
      k7 = a + one;
      k8 = a + two;
      k26update = -one;
      x = x / (one - x);
    }

    pkm2 = zero;
    qkm2 = one;
    pkm1 = one;
    qkm1 = one;
    ans = one;
    n = 0;

    do {
      xk = -(x * k1 * k2) / (k3 * k4);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      xk = (x * k5 * k6) / (k7 * k8);
      pk = pkm1 + pkm2 * xk;
      qk = qkm1 + qkm2 * xk;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;

      if (qk != zero) {
        r = pk / qk;
        if (std::abs(ans - r) < std::abs(r) * thresh) {
          return r;
        }
        ans = r;
      }

      k1 += one;
      k2 += k26update;
      k3 += two;
      k4 += two;
      k5 += one;
      k6 -= k26update;
      k7 += two;
      k8 += two;

      if ((std::abs(qk) + std::abs(pk)) > big) {
        pkm2 *= biginv;
        pkm1 *= biginv;
        qkm2 *= biginv;
        qkm1 *= biginv;
      }
      if ((std::abs(qk) < biginv) || (std::abs(pk) < biginv)) {
        pkm2 *= big;
        pkm1 *= big;
        qkm2 *= big;
        qkm1 *= big;
      }
    } while (++n < num_iters);

    return ans;
  }
};

/* Helper functions depending on the Scalar type */
template <typename Scalar>
struct betainc_helper {};

template <>
struct betainc_helper<float> {
  /* Core implementation, assumes a large (> 1.0) */
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE float incbsa(float aa, float bb,
                                                            float xx) {
    float ans, a, b, t, x, onemx;
    bool reversed_a_b = false;

    onemx = 1.0f - xx;

    /* see if x is greater than the mean */
    if (xx > (aa / (aa + bb))) {
      reversed_a_b = true;
      a = bb;
      b = aa;
      t = xx;
      x = onemx;
    } else {
      a = aa;
      b = bb;
      t = onemx;
      x = xx;
    }

    /* Choose expansion for optimal convergence */
    if (b > 10.0f) {
      if (std::abs(b * x / a) < 0.3f) {
        t = betainc_helper<float>::incbps(a, b, x);
        if (reversed_a_b) t = 1.0f - t;
        return t;
      }
    }

    ans = x * (a + b - 2.0f) / (a - 1.0f);
    if (ans < 1.0f) {
      ans = incbeta_cfe<float>::run(a, b, x, true /* small_branch */);
      t = b * std::log(t);
    } else {
      ans = incbeta_cfe<float>::run(a, b, x, false /* small_branch */);
      t = (b - 1.0f) * std::log(t);
    }

    t += a * std::log(x) + std::lgamma(a + b) -
         std::lgamma(a) - std::lgamma(b);
    t += std::log(ans / a);
    t = std::exp(t);

    if (reversed_a_b) t = 1.0f - t;
    return t;
  }

  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE float incbps(float a, float b, float x) {
    float t, u, y, s;
    const float machep = cephes_helper<float>::machep();

    y = a * std::log(x) + (b - 1.0f) * std::log1p(-x) - std::log(a);
    y -= std::lgamma(a) + std::lgamma(b);
    y += std::lgamma(a + b);

    t = x / (1.0f - x);
    s = 0.0f;
    u = 1.0f;
    do {
      b -= 1.0f;
      if (b == 0.0f) {
        break;
      }
      a += 1.0f;
      u *= t * b / a;
      s += u;
    } while (std::abs(u) > machep);

    return std::exp(y) * (1.0f + s);
  }
};

template <>
struct betainc_impl<float> {
  EIGEN_DEVICE_FUNC
  static float run(float a, float b, float x) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    float ans, t;

    if (a <= 0.0f) return nan;
    if (b <= 0.0f) return nan;
    if ((x <= 0.0f) || (x >= 1.0f)) {
      if (x == 0.0f) return 0.0f;
      if (x == 1.0f) return 1.0f;
      // mtherr("betaincf", DOMAIN);
      return nan;
    }

    /* transformation for small aa */
    if (a <= 1.0f) {
      ans = betainc_helper<float>::incbsa(a + 1.0f, b, x);
      t = a * std::log(x) + b * std::log1p(-x) +
          std::lgamma(a + b) - std::lgamma(a + 1.0f) -
          std::lgamma(b);
      return (ans + std::exp(t));
    } else {
      return betainc_helper<float>::incbsa(a, b, x);
    }
  }
};

template <>
struct betainc_helper<double> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE double incbps(double a, double b, double x) {
    const double machep = cephes_helper<double>::machep();

    double s, t, u, v, n, t1, z, ai;

    ai = 1.0 / a;
    u = (1.0 - b) * x;
    v = u / (a + 1.0);
    t1 = v;
    t = u;
    n = 2.0;
    s = 0.0;
    z = machep * ai;
    while (std::abs(v) > z) {
      u = (n - b) * x / n;
      t *= u;
      v = t / (a + n);
      s += v;
      n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * std::log(x);
    // TODO: gamma() is not directly implemented in Eigen.
    /*
    if ((a + b) < maxgam && std::abs(u) < maxlog) {
      t = gamma(a + b) / (gamma(a) * gamma(b));
      s = s * t * pow(x, a);
    }
    */
    t = std::lgamma(a + b) - std::lgamma(a) -
        std::lgamma(b) + u + std::log(s);
    return s = std::exp(t);
  }
};

template <>
struct betainc_impl<double> {
  EIGEN_DEVICE_FUNC
  static double run(double aa, double bb, double xx) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double machep = cephes_helper<double>::machep();
    // const double maxgam = 171.624376956302725;

    double a, b, t, x, xc, w, y;
    bool reversed_a_b = false;

    if (aa <= 0.0 || bb <= 0.0) {
      return nan;  // goto domerr;
    }

    if ((xx <= 0.0) || (xx >= 1.0)) {
      if (xx == 0.0) return (0.0);
      if (xx == 1.0) return (1.0);
      // mtherr("incbet", DOMAIN);
      return nan;
    }

    if ((bb * xx) <= 1.0 && xx <= 0.95) {
      return betainc_helper<double>::incbps(aa, bb, xx);
    }

    w = 1.0 - xx;

    /* Reverse a and b if x is greater than the mean. */
    if (xx > (aa / (aa + bb))) {
      reversed_a_b = true;
      a = bb;
      b = aa;
      xc = xx;
      x = w;
    } else {
      a = aa;
      b = bb;
      xc = w;
      x = xx;
    }

    if (reversed_a_b && (b * x) <= 1.0 && x <= 0.95) {
      t = betainc_helper<double>::incbps(a, b, x);
      if (t <= machep) {
        t = 1.0 - machep;
      } else {
        t = 1.0 - t;
      }
      return t;
    }

    /* Choose expansion for better convergence. */
    y = x * (a + b - 2.0) - (a - 1.0);
    if (y < 0.0) {
      w = incbeta_cfe<double>::run(a, b, x, true /* small_branch */);
    } else {
      w = incbeta_cfe<double>::run(a, b, x, false /* small_branch */) / xc;
    }

    /* Multiply w by the factor
         a      b   _             _     _
        x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

    y = a * std::log(x);
    t = b * std::log(xc);
    // TODO: gamma is not directly implemented in Eigen.
    /*
    if ((a + b) < maxgam && numext::abs(y) < maxlog && numext::abs(t) < maxlog)
    {
      t = pow(xc, b);
      t *= pow(x, a);
      t /= a;
      t *= w;
      t *= gamma(a + b) / (gamma(a) * gamma(b));
    } else {
    */
    /* Resort to logarithms.  */
    y += t + std::lgamma(a + b) - std::lgamma(a) -
         std::lgamma(b);
    y += std::log(w / a);
    t = std::exp(y);

    /* } */
    // done:

    if (reversed_a_b) {
      if (t <= machep) {
        t = 1.0 - machep;
      } else {
        t = 1.0 - t;
      }
    }
    return t;
  }
};

/* end of third_party/eigen/unsupported/Eigen/src/SpecialFunctions/SpecialFunctionsImpl.h */

template <typename T>
inline C10_HOST_DEVICE T calc_betainc(T a, T b, T x) {
    if (std::is_same_v<T, float>) {
        const float _x = x, _a = a, _b = b;
        return betainc_impl<T>::run(_a, _b, _x);
    } else if (std::is_same_v<T, double>) {
        const double _x = x, _a = a, _b = b;
        return betainc_impl<T>::run(_a, _b, _x);
    } else {
        //promote to float
        const float _x = static_cast<float>(x);
        const float _a = static_cast<float>(a);
        const float _b = static_cast<float>(b);
        return static_cast<T>(betainc_impl<float>::run(_a,_b,_x));
    }
}

} // namespace at::native
