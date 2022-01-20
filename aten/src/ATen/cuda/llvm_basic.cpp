// This file is modified from LLVM, see the following copyright information
//
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include <ATen/jit_macros.h>

#ifdef USE_JITERATOR

namespace at {
namespace cuda {

extern const std::string traits = R"ESCAPE(

namespace std {

template <class _Tp>
_Tp&& __declval(int);
template <class _Tp>
_Tp __declval(long);
template <class _Tp>
decltype(__declval<_Tp>(0)) declval() noexcept;

template <class _Tp, _Tp __v>
struct integral_constant {
  static const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
};

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

// is_same, functional
template <class _Tp, class _Up> struct is_same : public false_type {};
template <class _Tp> struct is_same<_Tp, _Tp> : public true_type {};

// is_integral, for some types.
template <class _Tp> struct is_integral
    : public integral_constant<bool, false> {};
template <> struct is_integral<bool>
    : public integral_constant<bool, true> {};
template <> struct is_integral<char>
    : public integral_constant<bool, true> {};
template <> struct is_integral<short>
    : public integral_constant<bool, true> {};
template <> struct is_integral<int>
    : public integral_constant<bool, true> {};
template <> struct is_integral<long>
    : public integral_constant<bool, true> {};
template <> struct is_integral<long long>
    : public integral_constant<bool, true> {};

// enable_if, functional
template <bool _C, typename _Tp> struct enable_if{};
template <typename _Tp> struct enable_if<true, _Tp>{
  using type = _Tp;
};
template <bool b, class T=void>
using enable_if_t = typename enable_if<b,T>::type;

template <class _Tp> struct remove_const            {typedef _Tp type;};
template <class _Tp> struct remove_const<const _Tp> {typedef _Tp type;};
template <class _Tp> using remove_const_t = typename remove_const<_Tp>::type;

template <class _Tp> struct remove_volatile               {typedef _Tp type;};
template <class _Tp> struct remove_volatile<volatile _Tp> {typedef _Tp type;};
template <class _Tp> using remove_volatile_t = typename remove_volatile<_Tp>::type;

template <class _Tp> struct remove_cv
{typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;};
template <class _Tp> using remove_cv_t = typename remove_cv<_Tp>::type;

template <class _Tp> struct __libcpp_is_floating_point              : public false_type {};
template <>          struct __libcpp_is_floating_point<float>       : public true_type {};
template <>          struct __libcpp_is_floating_point<double>      : public true_type {};
template <>          struct __libcpp_is_floating_point<long double> : public true_type {};

template <class _Tp> struct is_floating_point
    : public __libcpp_is_floating_point<typename remove_cv<_Tp>::type> {};

template <class _Tp> struct is_arithmetic
    : public integral_constant<bool, is_integral<_Tp>::value      ||
                                     is_floating_point<_Tp>::value> {};
template <class _Tp>
inline constexpr bool is_arithmetic_v = is_arithmetic<_Tp>::value;

template <class _Tp>
struct __numeric_type
{
   static void __test(...);
   static float __test(float);
   static double __test(char);
   static double __test(int);
   static double __test(unsigned);
   static double __test(long);
   static double __test(unsigned long);
   static double __test(long long);
   static double __test(unsigned long long);
   static double __test(double);
   static long double __test(long double);

   typedef decltype(__test(declval<_Tp>())) type;
   static const bool value = !is_same<type, void>::value;
};

template <>
struct __numeric_type<void>
{
   static const bool value = true;
};

// __promote

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&
                 __numeric_type<_A2>::value &&
                 __numeric_type<_A3>::value>
class __promote_imp
{
public:
    static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
    typedef typename __promote_imp<_A3>::type __type3;
public:
    typedef decltype(__type1() + __type2() + __type3()) type;
    static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
public:
    typedef decltype(__type1() + __type2()) type;
    static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
    typedef typename __numeric_type<_A1>::type type;
    static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

} // namespace std

)ESCAPE";

const std::string &get_traits_string() {
    return traits;
}

extern const std::string cmath = R"ESCAPE(

namespace std {

using ::signbit;
using ::isfinite;
using ::isinf;
using ::isnan;

using ::abs;

using ::acos;
using ::acosf;
using ::asin;
using ::asinf;
using ::atan;
using ::atanf;
using ::atan2;
using ::atan2f;
using ::ceil;
using ::ceilf;
using ::cos;
using ::cosf;
using ::cosh;
using ::coshf;

using ::exp;
using ::expf;

using ::fabs;
using ::fabsf;
using ::floor;
using ::floorf;

using ::fmod;
using ::fmodf;

using ::frexp;
using ::frexpf;
using ::ldexp;
using ::ldexpf;

using ::log;
using ::logf;

using ::log10;
using ::log10f;
using ::modf;
using ::modff;

using ::pow;
using ::powf;

using ::sin;
using ::sinf;
using ::sinh;
using ::sinhf;

using ::sqrt;
using ::sqrtf;
using ::tan;
using ::tanf;

using ::tanh;
using ::tanhf;

using ::acosh;
using ::acoshf;
using ::asinh;
using ::asinhf;
using ::atanh;
using ::atanhf;
using ::cbrt;
using ::cbrtf;

using ::copysign;
using ::copysignf;

using ::erf;
using ::erff;
using ::erfc;
using ::erfcf;
using ::exp2;
using ::exp2f;
using ::expm1;
using ::expm1f;
using ::fdim;
using ::fdimf;
using ::fmaf;
using ::fma;
using ::fmax;
using ::fmaxf;
using ::fmin;
using ::fminf;
using ::hypot;
using ::hypotf;
using ::ilogb;
using ::ilogbf;
using ::lgamma;
using ::lgammaf;
using ::llrint;
using ::llrintf;
using ::llround;
using ::llroundf;
using ::log1p;
using ::log1pf;
using ::log2;
using ::log2f;
using ::logb;
using ::logbf;
using ::lrint;
using ::lrintf;
using ::lround;
using ::lroundf;

using ::nan;
using ::nanf;

using ::nearbyint;
using ::nearbyintf;
using ::nextafter;
using ::nextafterf;
using ::remainder;
using ::remainderf;
using ::remquo;
using ::remquof;
using ::rint;
using ::rintf;
using ::round;
using ::roundf;
using ::scalbln;
using ::scalblnf;
using ::scalbn;
using ::scalbnf;
using ::tgamma;
using ::tgammaf;
using ::trunc;
using ::truncf;

// TODO: why does the following code fail to compile?
// inline float       hypot(       float x,       float y,       float z ) { return sqrt(x*x + y*y + z*z); }
// inline double      hypot(      double x,      double y,      double z ) { return sqrt(x*x + y*y + z*z); }
// inline long double hypot( long double x, long double y, long double z ) { return sqrt(x*x + y*y + z*z); }

// template <class _A1, class _A2, class _A3>
// inline
// typename enable_if_t
// <
//     is_arithmetic<_A1>::value &&
//     is_arithmetic<_A2>::value &&
//     is_arithmetic<_A3>::value,
//     __promote<_A1, _A2, _A3>
// >::type
// hypot(_A1 __lcpp_x, _A2 __lcpp_y, _A3 __lcpp_z) noexcept
// {
//     typedef typename __promote<_A1, _A2, _A3>::type __result_type;
//     static_assert((!(is_same<_A1, __result_type>::value &&
//                      is_same<_A2, __result_type>::value &&
//                      is_same<_A3, __result_type>::value)), "");
//     return hypot((__result_type)__lcpp_x, (__result_type)__lcpp_y, (__result_type)__lcpp_z);
// }

template <typename _Fp>
constexpr
_Fp __lerp(_Fp __a, _Fp __b, _Fp __t) noexcept {
    if ((__a <= 0 && __b >= 0) || (__a >= 0 && __b <= 0))
        return __t * __b + (1 - __t) * __a;

    if (__t == 1) return __b;
    const _Fp __x = __a + __t * (__b - __a);
    if ((__t > 1) == (__b > __a))
        return __b < __x ? __x : __b;
    else
        return __x < __b ? __x : __b;
}

constexpr float
lerp(float __a, float __b, float __t)                   noexcept { return __lerp(__a, __b, __t); }

constexpr double
lerp(double __a, double __b, double __t)                noexcept { return __lerp(__a, __b, __t); }

constexpr long double
lerp(long double __a, long double __b, long double __t) noexcept { return __lerp(__a, __b, __t); }

template <class _A1, class _A2, class _A3>
inline
constexpr typename enable_if_t
<
    is_arithmetic<_A1>::value &&
    is_arithmetic<_A2>::value &&
    is_arithmetic<_A3>::value,
    __promote<_A1, _A2, _A3>
>::type
lerp(_A1 __a, _A2 __b, _A3 __t) noexcept
{
    typedef typename __promote<_A1, _A2, _A3>::type __result_type;
    static_assert(!(_IsSame<_A1, __result_type>::value &&
                    _IsSame<_A2, __result_type>::value &&
                    _IsSame<_A3, __result_type>::value));
    return __lerp((__result_type)__a, (__result_type)__b, (__result_type)__t);
}

} // namespace std

)ESCAPE";

const std::string &get_cmath_string() {
    return cmath;
}

}} // namespace at::cuda

#endif
