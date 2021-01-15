#pragma once

#include <cmath>

// Android NDK platform < 21 with libstdc++ has spotty C++11 support.
// Various hacks in this header allow the rest of the codebase to use
// standard APIs.
#if defined(__ANDROID__) && __ANDROID_API__ < 21 && defined(__GLIBCXX__)

namespace std {
  // Import double versions of these functions from the global namespace.
  using ::acosh;
  using ::asinh;
  using ::atanh;
  using ::erf;
  using ::erfc;
  using ::expm1;
  using ::lgamma;
  using ::log1p;
  using ::nearbyint;
  using ::round;
  using ::tgamma;
  using ::trunc;
  using ::truncf;

  // Define float versions the same way as more recent libstdc++
  inline float acosh(float x) { return __builtin_acoshf(x); }
  inline float asinh(float x) { return __builtin_asinhf(x); }
  inline float atanh(float x) { return __builtin_atanhf(x); }
  inline float copysign(float x, float y) { return __builtin_copysignf(x, y); }
  inline float erf(float x) { return __builtin_erff(x); }
  inline float erfc(float x) { return __builtin_erfcf(x); }
  inline float expm1(float x) { return __builtin_expm1f(x); }
  inline float fmax(float x, float y) { return __builtin_fmaxf(x, y); }
  inline float fmin(float x, float y) { return __builtin_fminf(x, y); }
  inline float lgamma(float x) { return __builtin_lgammaf(x); }
  inline float log1p(float x) { return __builtin_log1pf(x); }
  inline float nearbyint(float x) { return __builtin_nearbyintf(x); }
  inline float remainder(float x, float y) { return __builtin_remainderf(x, y); }
  inline float round(float x) { return __builtin_roundf(x); }
  inline float tgamma(float x) { return __builtin_tgammaf(x); }
  inline float trunc(float x) { return __builtin_truncf(x); }

  // __builtin_nexttoward isn't doesn't work.  It appears to try to
  // link against the global nexttoward function, which is not present
  // prior to API 18.  Just bail for now.
  inline float nexttoward(float x, long double y) {
    throw std::runtime_error("std::nexttoward is not present on older Android");
  }
  inline double nexttoward(double x, long double y) {
    throw std::runtime_error("std::nexttoward is not present on older Android");
  }

  // TODO: this function needs to be implemented and tested. Currently just throw an error.
  inline float hypot(float x, float y) {
    throw std::runtime_error("std::hypot is not implemented on older Android");
  }
  inline double hypot(double x, double y) {
    throw std::runtime_error("std::hypot is not implemented on older Android");
  }

  // TODO: this function needs to be implemented and tested. Currently just throw an error.
  inline float igamma(float x, float y) {
    throw std::runtime_error("igamma is not implemented on older Android");
  }
  inline double igamma(double x, double y) {
    throw std::runtime_error("igamma is not implemented on older Android");
  }
  inline float igammac(float x, float y) {
    throw std::runtime_error("igammac is not implemented on older Android");
  }
  inline double igammac(double x, double y) {
    throw std::runtime_error("igammac is not implemented on older Android");
  }

  // TODO: this function needs to be implemented and tested. Currently just throw an error.
  inline float nextafter(float x, float y) {
    throw std::runtime_error("std::nextafter is not implemented on older Android");
  }
  inline double nextafter(double x, double y) {
    throw std::runtime_error("std::nextafter is not implemented on older Android");
  }

   // TODO: this function needs to be implemented and tested. Currently just throw an error.
  inline float exp2(float x) {
    throw std::runtime_error("std::exp2 is not implemented on older Android");
  }
  inline double exp2(double x) {
    throw std::runtime_error("std::exp2 is not implemented on older Android");
  }

  // Define integral versions the same way as more recent libstdc++
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type acosh(T x) { return __builtin_acosh(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type asinh(T x) { return __builtin_asinh(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type atanh(T x) { return __builtin_atanh(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type erf(T x) { return __builtin_erf(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type erfc(T x) { return __builtin_erfc(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type expm1(T x) { return __builtin_expm1(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type lgamma(T x) { return __builtin_lgamma(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type log1p(T x) { return __builtin_log1p(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type nearbyint(T x) { return __builtin_nearbyint(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type round(T x) { return __builtin_round(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type tgamma(T x) { return __builtin_tgamma(x); }
  template<typename T> typename std::enable_if<std::is_integral<T>::value, double>::type trunc(T x) { return __builtin_trunc(x); }

  // Convoluted definition of these binary functions for overloads other than
  // (float,float) and (double,double).  Using a template from __gnu_cxx
  // is dirty, but this code is only enabled on a dead platform, so there
  // shouldn't be any risk of it breaking due to updates.
  template<typename T, typename U>
  typename __gnu_cxx::__promote_2<T, U>::__type
  fmax(T x, U y) {
    typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
    return fmax(type(x), type(y));
  }
  template<typename T, typename U>
  typename __gnu_cxx::__promote_2<T, U>::__type
  fmin(T x, U y) {
    typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
    return fmin(type(x), type(y));
  }
  template<typename T, typename U>
  typename __gnu_cxx::__promote_2<T, U>::__type
  copysign(T x, U y) {
    typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
    return copysign(type(x), type(y));
  }
  template<typename T, typename U>
  typename __gnu_cxx::__promote_2<T, U>::__type
  remainder(T x, U y) {
    typedef typename __gnu_cxx::__promote_2<T, U>::__type type;
    return remainder(type(x), type(y));
  }

  // log2 is a macro on Android API < 21, so we need to define it ourselves.
  inline float log2(float arg) {
    return ::log(arg) / ::log(2.0);
  }
  inline double log2(double arg) {
    return ::log(arg) / ::log(2.0);
  }
  inline long double log2(long double arg) {
    return ::log(arg) / ::log(2.0);
  }
  template<typename T>
  typename std::enable_if<std::is_integral<T>::value, double>::type
  log2(T x) {
    return ::log(x) / ::log(2.0);
  }
}

#endif
