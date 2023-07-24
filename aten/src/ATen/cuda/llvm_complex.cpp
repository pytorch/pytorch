// This is copy-pasted (with modification) from the following llvm file:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/complex
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
#include <ATen/cuda/llvm_jit_strings.h>


namespace at::cuda {

const std::string complex_body = R"ESCAPE(

namespace std {

template<class _Tp> class complex;

template<class _Tp> complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w);
template<class _Tp> complex<_Tp> operator/(const complex<_Tp>& __x, const complex<_Tp>& __y);

template<class _Tp>
class complex
{
public:
    typedef _Tp value_type;
private:
    value_type __re_;
    value_type __im_;
public:
    constexpr
    complex(const value_type& __re = value_type(), const value_type& __im = value_type())
        : __re_(__re), __im_(__im) {}
    template<class _Xp> constexpr
    complex(const complex<_Xp>& __c)
        : __re_(__c.real()), __im_(__c.imag()) {}

    constexpr value_type real() const {return __re_;}
    constexpr value_type imag() const {return __im_;}

    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    constexpr operator bool() const {
        return real() || imag();
    }

    complex& operator= (const value_type& __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    complex& operator+=(const value_type& __re) {__re_ += __re; return *this;}
    complex& operator-=(const value_type& __re) {__re_ -= __re; return *this;}
    complex& operator*=(const value_type& __re) {__re_ *= __re; __im_ *= __re; return *this;}
    complex& operator/=(const value_type& __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

template<> class complex<double>;

template<>
class complex<float>
{
    float __re_;
    float __im_;
public:
    typedef float value_type;

    constexpr complex(float __re = 0.0f, float __im = 0.0f)
        : __re_(__re), __im_(__im) {}

    explicit constexpr complex(const complex<double>& __c);

    constexpr float real() const {return __re_;}
    constexpr float imag() const {return __im_;}

    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    constexpr operator bool() const {
        return real() || imag();
    }

    complex& operator= (float __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    complex& operator+=(float __re) {__re_ += __re; return *this;}
    complex& operator-=(float __re) {__re_ -= __re; return *this;}
    complex& operator*=(float __re) {__re_ *= __re; __im_ *= __re; return *this;}
    complex& operator/=(float __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

template<>
class complex<double>
{
    double __re_;
    double __im_;
public:
    typedef double value_type;

    constexpr complex(double __re = 0.0, double __im = 0.0)
        : __re_(__re), __im_(__im) {}

    constexpr complex(const complex<float>& __c);

    constexpr double real() const {return __re_;}
    constexpr double imag() const {return __im_;}

    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    constexpr operator bool() const {
        return real() || imag();
    }

    complex& operator= (double __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    complex& operator+=(double __re) {__re_ += __re; return *this;}
    complex& operator-=(double __re) {__re_ -= __re; return *this;}
    complex& operator*=(double __re) {__re_ *= __re; __im_ *= __re; return *this;}
    complex& operator/=(double __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

inline
constexpr
complex<float>::complex(const complex<double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline
constexpr
complex<double>::complex(const complex<float>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}


// 26.3.6 operators:

template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator+(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t += __x;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(-__y);
    __t += __x;
    return __t;
}

template<class _Tp>
complex<_Tp>
operator*(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __ac = __a * __c;
    _Tp __bd = __b * __d;
    _Tp __ad = __a * __d;
    _Tp __bc = __b * __c;
    _Tp __x = __ac - __bd;
    _Tp __y = __ad + __bc;
    if (isnan(__x) && isnan(__y))
    {
        bool __recalc = false;
        if (isinf(__a) || isinf(__b))
        {
            __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
            __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
            if (isnan(__c))
                __c = copysign(_Tp(0), __c);
            if (isnan(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (isinf(__c) || isinf(__d))
        {
            __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
            __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
            if (isnan(__a))
                __a = copysign(_Tp(0), __a);
            if (isnan(__b))
                __b = copysign(_Tp(0), __b);
            __recalc = true;
        }
        if (!__recalc && (isinf(__ac) || isinf(__bd) ||
                          isinf(__ad) || isinf(__bc)))
        {
            if (isnan(__a))
                __a = copysign(_Tp(0), __a);
            if (isnan(__b))
                __b = copysign(_Tp(0), __b);
            if (isnan(__c))
                __c = copysign(_Tp(0), __c);
            if (isnan(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (__recalc)
        {
            __x = _Tp(INFINITY) * (__a * __c - __b * __d);
            __y = _Tp(INFINITY) * (__a * __d + __b * __c);
        }
    }
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline
complex<_Tp>
operator*(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t *= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator*(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t *= __x;
    return __t;
}

template<class _Tp>
complex<_Tp>
operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    int __ilogbw = 0;
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __logbw = logb(fmax(fabs(__c), fabs(__d)));
    if (isfinite(__logbw))
    {
        __ilogbw = static_cast<int>(__logbw);
        __c = scalbn(__c, -__ilogbw);
        __d = scalbn(__d, -__ilogbw);
    }
    _Tp __denom = __c * __c + __d * __d;
    _Tp __x = scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    _Tp __y = scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (isnan(__x) && isnan(__y))
    {
        if ((__denom == _Tp(0)) && (!isnan(__a) || !isnan(__b)))
        {
            __x = copysign(_Tp(INFINITY), __c) * __a;
            __y = copysign(_Tp(INFINITY), __c) * __b;
        }
        else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d))
        {
            __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
            __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
            __x = _Tp(INFINITY) * (__a * __c + __b * __d);
            __y = _Tp(INFINITY) * (__b * __c - __a * __d);
        }
        else if (isinf(__logbw) && __logbw > _Tp(0) && isfinite(__a) && isfinite(__b))
        {
            __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
            __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
            __x = _Tp(0) * (__a * __c + __b * __d);
            __y = _Tp(0) * (__b * __c - __a * __d);
        }
    }
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline
complex<_Tp>
operator/(const complex<_Tp>& __x, const _Tp& __y)
{
    return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template<class _Tp>
inline
complex<_Tp>
operator/(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t /= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x)
{
    return __x;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x)
{
    return complex<_Tp>(-__x.real(), -__x.imag());
}

template<class _Tp>
inline constexpr
bool
operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template<class _Tp>
inline constexpr
bool
operator==(const complex<_Tp>& __x, const _Tp& __y)
{
    return __x.real() == __y && __x.imag() == 0;
}

template<class _Tp>
inline constexpr
bool
operator==(const _Tp& __x, const complex<_Tp>& __y)
{
    return __x == __y.real() && 0 == __y.imag();
}

template<class _Tp>
inline constexpr
bool
operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator&&(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return bool(__x) && bool(__y);
}

template<class _Tp>
inline constexpr
bool
isnan(const complex<_Tp>& __x)
{
    return isnan(__x.real()) || isnan(__x.imag());
}

template<class _Tp>
inline constexpr
bool
operator||(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return bool(__x) || bool(__y);
}

// 26.3.7 values:

template <class _Tp, bool = is_integral<_Tp>::value,
                     bool = is_floating_point<_Tp>::value
                     >
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, true, false>
{
    typedef double _ValueType;
    typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, false, true>
{
    typedef _Tp _ValueType;
    typedef complex<_Tp> _ComplexType;
};

// real

template<class _Tp>
inline constexpr
_Tp
real(const complex<_Tp>& __c)
{
    return __c.real();
}

template <class _Tp>
inline constexpr
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
real(_Tp __re)
{
    return __re;
}

// imag

template<class _Tp>
inline constexpr
_Tp
imag(const complex<_Tp>& __c)
{
    return __c.imag();
}

template <class _Tp>
inline constexpr
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
imag(_Tp)
{
    return 0;
}

// abs

template<class _Tp>
inline
_Tp
abs(const complex<_Tp>& __c)
{
    return hypot(__c.real(), __c.imag());
}

// arg

template<class _Tp>
inline
_Tp
arg(const complex<_Tp>& __c)
{
    return atan2(__c.imag(), __c.real());
}

template<class _Tp>
inline
typename enable_if
<
    is_integral<_Tp>::value || is_same<_Tp, double>::value,
    double
>::type
arg(_Tp __re)
{
    return atan2(0., __re);
}

template <class _Tp>
inline
typename enable_if<
    is_same<_Tp, float>::value,
    float
>::type
arg(_Tp __re)
{
    return atan2f(0.F, __re);
}

}

)ESCAPE";

const std::string complex_half_body = R"ESCAPE(
namespace std {
template <>
struct alignas(2) complex<at::Half> {
  at::Half real_;
  at::Half imag_;

  // Constructors
  complex() = default;

  // implicit casting to and from `complex<float>`.
  // NOTE: computation of `complex<Half>` will occur in `complex<float>`
  __host__ __device__ inline complex(const std::complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  inline __host__ __device__ operator std::complex<float>() const {
    return {real_, imag_};
  }

  at::Half real() const {return real_;}
  at::Half imag() const {return imag_;}

};
}
)ESCAPE";


const std::string &get_complex_body_string() {
  return complex_body;
}

const std::string &get_complex_half_body_string() {
  return complex_half_body;
}

const std::string complex_math = R"ESCAPE(

namespace std {

// norm

template<class _Tp>
inline
_Tp
norm(const complex<_Tp>& __c)
{
    if (isinf(__c.real()))
        return abs(__c.real());
    if (isinf(__c.imag()))
        return abs(__c.imag());
    return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
inline
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
norm(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
    return static_cast<_ValueType>(__re) * __re;
}

// conj

template<class _Tp>
inline
complex<_Tp>
conj(const complex<_Tp>& __c)
{
    return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
inline
typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
conj(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}



// proj

template<class _Tp>
inline
complex<_Tp>
proj(const complex<_Tp>& __c)
{
    complex<_Tp> __r = __c;
    if (isinf(__c.real()) || isinf(__c.imag()))
        __r = complex<_Tp>(INFINITY, copysign(_Tp(0), __c.imag()));
    return __r;
}

template <class _Tp>
inline
typename enable_if
<
    is_floating_point<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    if (isinf(__re))
        __re = abs(__re);
    return complex<_Tp>(__re);
}

template <class _Tp>
inline
typename enable_if
<
    is_integral<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}

// polar

template<class _Tp>
complex<_Tp>
polar(const _Tp& __rho, const _Tp& __theta = _Tp())
{
    if (isnan(__rho) || signbit(__rho))
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    if (isnan(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, __theta);
        return complex<_Tp>(__theta, __theta);
    }
    if (isinf(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, _Tp(NAN));
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    }
    _Tp __x = __rho * cos(__theta);
    if (isnan(__x))
        __x = 0;
    _Tp __y = __rho * sin(__theta);
    if (isnan(__y))
        __y = 0;
    return complex<_Tp>(__x, __y);
}

// log

template<class _Tp>
inline
complex<_Tp>
log(const complex<_Tp>& __x)
{
    return complex<_Tp>(log(abs(__x)), arg(__x));
}

// log10

template<class _Tp>
inline
complex<_Tp>
log10(const complex<_Tp>& __x)
{
    return log(__x) / log(_Tp(10));
}

// log2

template<class _Tp>
inline
complex<_Tp>
log2(const complex<_Tp>& __x)
{
    return log(__x) / log(_Tp(2));
}

// sqrt

template<class _Tp>
complex<_Tp>
sqrt(const complex<_Tp>& __x)
{
    if (isinf(__x.imag()))
        return complex<_Tp>(_Tp(INFINITY), __x.imag());
    if (isinf(__x.real()))
    {
        if (__x.real() > _Tp(0))
            return complex<_Tp>(__x.real(), isnan(__x.imag()) ? __x.imag() : copysign(_Tp(0), __x.imag()));
        return complex<_Tp>(isnan(__x.imag()) ? __x.imag() : _Tp(0), copysign(__x.real(), __x.imag()));
    }
    return polar(sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template<class _Tp>
complex<_Tp>
exp(const complex<_Tp>& __x)
{
    _Tp __i = __x.imag();
    if (__i == 0) {
        return complex<_Tp>(exp(__x.real()), copysign(_Tp(0), __x.imag()));
    }
    if (isinf(__x.real()))
    {
        if (__x.real() < _Tp(0))
        {
            if (!isfinite(__i))
                __i = _Tp(1);
        }
        else if (__i == 0 || !isfinite(__i))
        {
            if (isinf(__i))
                __i = _Tp(NAN);
            return complex<_Tp>(__x.real(), __i);
        }
    }
    _Tp __e = exp(__x.real());
    return complex<_Tp>(__e * cos(__i), __e * sin(__i));
}

// pow

template<class _Tp>
inline
complex<_Tp>
pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return exp(__y * log(__x));
}

template<class _Tp, class _Up>
inline
complex<typename __promote<_Tp, _Up>::type>
pow(const complex<_Tp>& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

template<class _Tp, class _Up>
inline
typename enable_if
<
    is_arithmetic<_Up>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const complex<_Tp>& __x, const _Up& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

template<class _Tp, class _Up>
inline
typename enable_if
<
    is_arithmetic<_Tp>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const _Tp& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

// __sqr, computes pow(x, 2)

template<class _Tp>
inline
complex<_Tp>
__sqr(const complex<_Tp>& __x)
{
    return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                        _Tp(2) * __x.real() * __x.imag());
}

// asinh

template<class _Tp>
complex<_Tp>
asinh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return __x;
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (__x.imag() == 0)
            return __x;
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(copysign(__x.imag(), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) + _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// acosh

template<class _Tp>
complex<_Tp>
acosh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return complex<_Tp>(abs(__x.real()), __x.imag());
        if (isinf(__x.imag()))
        {
            if (__x.real() > 0)
                return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
            else
                return complex<_Tp>(-__x.real(), copysign(__pi * _Tp(0.75), __x.imag()));
        }
        if (__x.real() < 0)
            return complex<_Tp>(-__x.real(), copysign(__pi, __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(abs(__x.imag()), __x.real());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(abs(__x.imag()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), _Tp(0)), copysign(__z.imag(), __x.imag()));
}

// atanh

template<class _Tp>
complex<_Tp>
atanh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.imag()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (isnan(__x.imag()))
    {
        if (isinf(__x.real()) || __x.real() == 0)
            return complex<_Tp>(copysign(_Tp(0), __x.real()), __x.imag());
        return complex<_Tp>(__x.imag(), __x.imag());
    }
    if (isnan(__x.real()))
    {
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.real()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (abs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0))
    {
        return complex<_Tp>(copysign(_Tp(INFINITY), __x.real()), copysign(_Tp(0), __x.imag()));
    }
    complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// sinh

template<class _Tp>
complex<_Tp>
sinh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return __x;
    return complex<_Tp>(sinh(__x.real()) * cos(__x.imag()), cosh(__x.real()) * sin(__x.imag()));
}

// cosh

template<class _Tp>
complex<_Tp>
cosh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(abs(__x.real()), _Tp(NAN));
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(_Tp(NAN), __x.real());
    if (__x.real() == 0 && __x.imag() == 0)
        return complex<_Tp>(_Tp(1), __x.imag());
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return complex<_Tp>(abs(__x.real()), __x.imag());
    return complex<_Tp>(cosh(__x.real()) * cos(__x.imag()), sinh(__x.real()) * sin(__x.imag()));
}

// tanh

template<class _Tp>
complex<_Tp>
tanh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()))
    {
        if (!isfinite(__x.imag()))
            return complex<_Tp>(copysign(_Tp(1), __x.real()), _Tp(0));
        return complex<_Tp>(copysign(_Tp(1), __x.real()), copysign(_Tp(0), sin(_Tp(2) * __x.imag())));
    }
    if (isnan(__x.real()) && __x.imag() == 0)
        return __x;
    _Tp __2r(_Tp(2) * __x.real());
    _Tp __2i(_Tp(2) * __x.imag());
    _Tp __d(cosh(__2r) + cos(__2i));
    _Tp __2rsh(sinh(__2r));
    if (isinf(__2rsh) && isinf(__d))
        return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1),
                            __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
    return  complex<_Tp>(__2rsh/__d, sin(__2i)/__d);
}

// asin

template<class _Tp>
complex<_Tp>
asin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template<class _Tp>
complex<_Tp>
acos(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (isinf(__x.imag()))
        {
            if (__x.real() < _Tp(0))
                return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
            return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
        }
        if (__x.real() < _Tp(0))
            return complex<_Tp>(__pi, signbit(__x.imag()) ? -__x.real() : __x.real());
        return complex<_Tp>(_Tp(0), signbit(__x.imag()) ? __x.real() : -__x.real());
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), -__x.imag());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    if (__x.real() == 0 && (__x.imag() == 0 || isnan(__x.imag())))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
    if (signbit(__x.imag()))
        return complex<_Tp>(abs(__z.imag()), abs(__z.real()));
    return complex<_Tp>(abs(__z.imag()), -abs(__z.real()));
}

// atan

template<class _Tp>
complex<_Tp>
atan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template<class _Tp>
complex<_Tp>
sin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template<class _Tp>
inline
complex<_Tp>
cos(const complex<_Tp>& __x)
{
    return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template<class _Tp>
complex<_Tp>
tan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// Literal suffix for complex number literals [complex.literals]
inline namespace literals
{
  inline namespace complex_literals
  {
    constexpr complex<double> operator""i(long double __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }

    constexpr complex<double> operator""i(unsigned long long __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }


    constexpr complex<float> operator""if(long double __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }

    constexpr complex<float> operator""if(unsigned long long __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }
  } // namespace complex_literals
} // namespace literals

} // namespace std

#define FLT_EPSILON 1.192092896e-07F
#define DBL_EPSILON 2.2204460492503131e-016
#define LDBL_EPSILON 2.2204460492503131e-016

#define FLT_MAX 3.402823466e+38F
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MAX 1.7976931348623158e+308

#define FLT_MAX_EXP 128
#define DBL_MAX_EXP 1024
#define LDBL_MAX_EXP 1024

#define FLT_MANT_DIG 24
#define DBL_MANT_DIG 53
#define LDBL_MANT_DIG 53

//thrust/detail/complex/math_private.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* adapted from FreeBSD:
 *    lib/msun/src/math_private.h
 */
#pragma once



namespace thrust{
using std::complex;
namespace detail{
namespace complex{

using std::complex;

typedef union
{
  float value;
  uint32_t word;
} ieee_float_shape_type;
  
__host__ __device__
inline void get_float_word(uint32_t & i, float d){
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

__host__ __device__
inline void get_float_word(int32_t & i, float d){
  ieee_float_shape_type gf_u;
  gf_u.value = (d);
  (i) = gf_u.word;
}

__host__ __device__
inline void set_float_word(float & d, uint32_t i){
  ieee_float_shape_type sf_u;
  sf_u.word = (i);
  (d) = sf_u.value;
}

// Assumes little endian ordering
typedef union
{
  double value;
  struct
  {
    uint32_t lsw;
    uint32_t msw;
  } parts;
  struct
  {
    uint64_t w;
  } xparts;
} ieee_double_shape_type;
  
__host__ __device__ inline
void get_high_word(uint32_t & i,double d){
  ieee_double_shape_type gh_u;
  gh_u.value = (d);
  (i) = gh_u.parts.msw;                                   
}
  
/* Set the more significant 32 bits of a double from an int.  */
__host__ __device__ inline
void set_high_word(double & d, uint32_t v){
  ieee_double_shape_type sh_u;
  sh_u.value = (d);
  sh_u.parts.msw = (v);
  (d) = sh_u.value;
}
  
  
__host__ __device__ inline 
void  insert_words(double & d, uint32_t ix0, uint32_t ix1){
  ieee_double_shape_type iw_u;
  iw_u.parts.msw = (ix0);
  iw_u.parts.lsw = (ix1);
  (d) = iw_u.value;
}
  
/* Get two 32 bit ints from a double.  */
__host__ __device__ inline
void  extract_words(uint32_t & ix0,uint32_t & ix1, double d){
  ieee_double_shape_type ew_u;
  ew_u.value = (d);
  (ix0) = ew_u.parts.msw;
  (ix1) = ew_u.parts.lsw;
}
  
/* Get two 32 bit ints from a double.  */
__host__ __device__ inline
void  extract_words(int32_t & ix0,int32_t & ix1, double d){
  ieee_double_shape_type ew_u;
  ew_u.value = (d);
  (ix0) = ew_u.parts.msw;
  (ix1) = ew_u.parts.lsw;
}
  
} // namespace complex

} // namespace detail

} // namespace thrust


//thrust/detail/complex/math_private.h

//thrust/detail/complex/c99math.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#pragma once


namespace thrust{
using std::complex;
namespace detail
{
namespace complex
{

// Define basic arithmetic functions so we can use them without explicit scope
// keeping the code as close as possible to FreeBSDs for ease of maintenance.
// It also provides an easy way to support compilers with missing C99 functions.
// When possible, just use the names in the global scope.
// Some platforms define these as macros, others as free functions.
// Avoid using the std:: form of these as nvcc may treat std::foo() as __host__ functions.

using ::log;
using ::acos;
using ::asin;
using ::sqrt;
using ::sinh;
using ::tan;
using ::cos;
using ::sin;
using ::exp;
using ::cosh;
using ::atan;

template <typename T>
inline __host__ __device__ T infinity();

template <>
inline __host__ __device__ float infinity<float>()
{
  float res;
  set_float_word(res, 0x7f800000);
  return res;
}


template <>
inline __host__ __device__ double infinity<double>()
{
  double res;
  insert_words(res, 0x7ff00000,0);
  return res;
}

#if defined _MSC_VER
__host__ __device__ inline int isinf(float x){
  return std::abs(x) == infinity<float>();
}

__host__ __device__ inline int isinf(double x){
  return std::abs(x) == infinity<double>();
}

__host__ __device__ inline int isnan(float x){
  return x != x;
}

__host__ __device__ inline int isnan(double x){
  return x != x;
}

__host__ __device__ inline int signbit(float x){
  return ((*((uint32_t *)&x)) & 0x80000000) != 0 ? 1 : 0;
}

__host__ __device__ inline int signbit(double x){
  return ((*((uint64_t *)&x)) & 0x8000000000000000) != 0ull ? 1 : 0;
}

__host__ __device__ inline int isfinite(float x){
  return !isnan(x) && !isinf(x);
}

__host__ __device__ inline int isfinite(double x){
  return !isnan(x) && !isinf(x);
}

#else

#  if defined(__CUDACC__) && !(defined(__CUDA__) && defined(__clang__)) && !defined(_NVHPC_CUDA)
// NVCC implements at least some signature of these as functions not macros.
using ::isinf;
using ::isnan;
using ::signbit;
using ::isfinite;
#  else
// Some compilers do not provide these in the global scope, because they are
// supposed to be macros. The versions in `std` are supposed to be functions.
// Since we're not compiling with nvcc, it's safe to use the functions in std::
using std::isinf;
using std::isnan;
using std::signbit;
using std::isfinite;
#  endif // __CUDACC__
#endif // _MSC_VER

using ::atanh;

#if defined _MSC_VER

__host__ __device__ inline double copysign(double x, double y){
  uint32_t hx,hy;
  get_high_word(hx,x);
  get_high_word(hy,y);
  set_high_word(x,(hx&0x7fffffff)|(hy&0x80000000));
  return x;
}

__host__ __device__ inline float copysignf(float x, float y){
  uint32_t ix,iy;
  get_float_word(ix,x);
  get_float_word(iy,y);
  set_float_word(x,(ix&0x7fffffff)|(iy&0x80000000));
  return x;
}



#if !defined(__CUDACC__) && !defined(_NVHPC_CUDA)

// Simple approximation to log1p as Visual Studio is lacking one
inline double log1p(double x){
  double u = 1.0+x;
  if(u == 1.0){
    return x;
  }else{
    if(u > 2.0){
      // Use normal log for large arguments
      return log(u);
    }else{
      return log(u)*(x/(u-1.0));
    }
  }
}

inline float log1pf(float x){
  float u = 1.0f+x;
  if(u == 1.0f){
    return x;
  }else{
    if(u > 2.0f){
      // Use normal log for large arguments
      return logf(u);
    }else{
      return logf(u)*(x/(u-1.0f));
    }
  }
}

#if _MSV_VER <= 1500
// #includec <complex>

inline float hypotf(float x, float y){
	return abs(std::complex<float>(x,y));
}

inline double hypot(double x, double y){
	return _hypot(x,y);
}

#endif // _MSC_VER <= 1500

#endif // __CUDACC__

#endif // _MSC_VER

} // namespace complex

} // namespace detail

} // namespace thrust

//thrust/detail/complex/c99math.h

//thrust/detail/complex/catrig.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia <filipe.c.maia@gmail.com>:
 *    freebsd/lib/msun/src/catrig.c
 */

#pragma once


namespace thrust{
using std::complex;
namespace detail{
namespace complex{

using std::complex;

__host__ __device__
inline void raise_inexact(){
  const volatile float tiny = 7.888609052210118054117286e-31; /* 0x1p-100; */
  // needs the volatile to prevent compiler from ignoring it
  volatile float junk = 1 + tiny;
  (void)junk;
}

__host__ __device__ inline complex<double> clog_for_large_values(complex<double> z);

/*
 * Testing indicates that all these functions are accurate up to 4 ULP.
 * The functions casin(h) and cacos(h) are about 2.5 times slower than asinh.
 * The functions catan(h) are a little under 2 times slower than atanh.
 *
 * The code for casinh, casin, cacos, and cacosh comes first.  The code is
 * rather complicated, and the four functions are highly interdependent.
 *
 * The code for catanh and catan comes at the end.  It is much simpler than
 * the other functions, and the code for these can be disconnected from the
 * rest of the code.
 */

/*
 *			================================
 *			| casinh, casin, cacos, cacosh |
 *			================================
 */

/*
 * The algorithm is very close to that in "Implementing the complex arcsine
 * and arccosine functions using exception handling" by T. E. Hull, Thomas F.
 * Fairgrieve, and Ping Tak Peter Tang, published in ACM Transactions on
 * Mathematical Software, Volume 23 Issue 3, 1997, Pages 299-335,
 * http://dl.acm.org/citation.cfm?id=275324.
 *
 * Throughout we use the convention z = x + I*y.
 *
 * casinh(z) = sign(x)*log(A+sqrt(A*A-1)) + I*asin(B)
 * where
 * A = (|z+I| + |z-I|) / 2
 * B = (|z+I| - |z-I|) / 2 = y/A
 *
 * These formulas become numerically unstable:
 *   (a) for Re(casinh(z)) when z is close to the line segment [-I, I] (that
 *       is, Re(casinh(z)) is close to 0);
 *   (b) for Im(casinh(z)) when z is close to either of the intervals
 *       [I, I*infinity) or (-I*infinity, -I] (that is, |Im(casinh(z))| is
 *       close to PI/2).
 *
 * These numerical problems are overcome by defining
 * f(a, b) = (hypot(a, b) - b) / 2 = a*a / (hypot(a, b) + b) / 2
 * Then if A < A_crossover, we use
 *   log(A + sqrt(A*A-1)) = log1p((A-1) + sqrt((A-1)*(A+1)))
 *   A-1 = f(x, 1+y) + f(x, 1-y)
 * and if B > B_crossover, we use
 *   asin(B) = atan2(y, sqrt(A*A - y*y)) = atan2(y, sqrt((A+y)*(A-y)))
 *   A-y = f(x, y+1) + f(x, y-1)
 * where without loss of generality we have assumed that x and y are
 * non-negative.
 *
 * Much of the difficulty comes because the intermediate computations may
 * produce overflows or underflows.  This is dealt with in the paper by Hull
 * et al by using exception handling.  We do this by detecting when
 * computations risk underflow or overflow.  The hardest part is handling the
 * underflows when computing f(a, b).
 *
 * Note that the function f(a, b) does not appear explicitly in the paper by
 * Hull et al, but the idea may be found on pages 308 and 309.  Introducing the
 * function f(a, b) allows us to concentrate many of the clever tricks in this
 * paper into one function.
 */

/*
 * Function f(a, b, hypot_a_b) = (hypot(a, b) - b) / 2.
 * Pass hypot(a, b) as the third argument.
 */
__host__ __device__
inline double
f(double a, double b, double hypot_a_b)
{
  if (b < 0)
    return ((hypot_a_b - b) / 2);
  if (b == 0)
    return (a / 2);
  return (a * a / (hypot_a_b + b) / 2);
}

/*
 * All the hard work is contained in this function.
 * x and y are assumed positive or zero, and less than RECIP_EPSILON.
 * Upon return:
 * rx = Re(casinh(z)) = -Im(cacos(y + I*x)).
 * B_is_usable is set to 1 if the value of B is usable.
 * If B_is_usable is set to 0, sqrt_A2my2 = sqrt(A*A - y*y), and new_y = y.
 * If returning sqrt_A2my2 has potential to result in an underflow, it is
 * rescaled, and new_y is similarly rescaled.
 */
__host__ __device__
inline void
do_hard_work(double x, double y, double *rx, int *B_is_usable, double *B,
			double *sqrt_A2my2, double *new_y)
{
  double R, S, A; /* A, B, R, and S are as in Hull et al. */
  double Am1, Amy; /* A-1, A-y. */
  const double A_crossover = 10; /* Hull et al suggest 1.5, but 10 works better */
  const double FOUR_SQRT_MIN = 5.966672584960165394632772e-154; /* =0x1p-509; >= 4 * sqrt(DBL_MIN) */
  const double B_crossover = 0.6417; /* suggested by Hull et al */

  R = hypot(x, y + 1);		/* |z+I| */
  S = hypot(x, y - 1);		/* |z-I| */

  /* A = (|z+I| + |z-I|) / 2 */
  A = (R + S) / 2;
  /*
   * Mathematically A >= 1.  There is a small chance that this will not
   * be so because of rounding errors.  So we will make certain it is
   * so.
   */
  if (A < 1)
    A = 1;

  if (A < A_crossover) {
    /*
     * Am1 = fp + fm, where fp = f(x, 1+y), and fm = f(x, 1-y).
     * rx = log1p(Am1 + sqrt(Am1*(A+1)))
     */
    if (y == 1 && x < DBL_EPSILON * DBL_EPSILON / 128) {
      /*
       * fp is of order x^2, and fm = x/2.
       * A = 1 (inexactly).
       */
      *rx = sqrt(x);
    } else if (x >= DBL_EPSILON * fabs(y - 1)) {
      /*
       * Underflow will not occur because
       * x >= DBL_EPSILON^2/128 >= FOUR_SQRT_MIN
       */
      Am1 = f(x, 1 + y, R) + f(x, 1 - y, S);
      *rx = log1p(Am1 + sqrt(Am1 * (A + 1)));
    } else if (y < 1) {
      /*
       * fp = x*x/(1+y)/4, fm = x*x/(1-y)/4, and
       * A = 1 (inexactly).
       */
      *rx = x / sqrt((1 - y) * (1 + y));
    } else {		/* if (y > 1) */
      /*
       * A-1 = y-1 (inexactly).
       */
      *rx = log1p((y - 1) + sqrt((y - 1) * (y + 1)));
    }
  } else {
    *rx = log(A + sqrt(A * A - 1));
  }

  *new_y = y;

  if (y < FOUR_SQRT_MIN) {
    /*
     * Avoid a possible underflow caused by y/A.  For casinh this
     * would be legitimate, but will be picked up by invoking atan2
     * later on.  For cacos this would not be legitimate.
     */
    *B_is_usable = 0;
    *sqrt_A2my2 = A * (2 / DBL_EPSILON);
    *new_y = y * (2 / DBL_EPSILON);
    return;
  }

  /* B = (|z+I| - |z-I|) / 2 = y/A */
  *B = y / A;
  *B_is_usable = 1;

  if (*B > B_crossover) {
    *B_is_usable = 0;
    /*
     * Amy = fp + fm, where fp = f(x, y+1), and fm = f(x, y-1).
     * sqrt_A2my2 = sqrt(Amy*(A+y))
     */
    if (y == 1 && x < DBL_EPSILON / 128) {
      /*
       * fp is of order x^2, and fm = x/2.
       * A = 1 (inexactly).
       */
      *sqrt_A2my2 = sqrt(x) * sqrt((A + y) / 2);
    } else if (x >= DBL_EPSILON * fabs(y - 1)) {
      /*
       * Underflow will not occur because
       * x >= DBL_EPSILON/128 >= FOUR_SQRT_MIN
       * and
       * x >= DBL_EPSILON^2 >= FOUR_SQRT_MIN
       */
      Amy = f(x, y + 1, R) + f(x, y - 1, S);
      *sqrt_A2my2 = sqrt(Amy * (A + y));
    } else if (y > 1) {
      /*
       * fp = x*x/(y+1)/4, fm = x*x/(y-1)/4, and
       * A = y (inexactly).
       *
       * y < RECIP_EPSILON.  So the following
       * scaling should avoid any underflow problems.
       */
      *sqrt_A2my2 = x * (4 / DBL_EPSILON / DBL_EPSILON) * y /
	sqrt((y + 1) * (y - 1));
      *new_y = y * (4 / DBL_EPSILON / DBL_EPSILON);
    } else {		/* if (y < 1) */
      /*
       * fm = 1-y >= DBL_EPSILON, fp is of order x^2, and
       * A = 1 (inexactly).
       */
      *sqrt_A2my2 = sqrt((1 - y) * (1 + y));
    }
  }
}

/*
 * casinh(z) = z + O(z^3)   as z -> 0
 *
 * casinh(z) = sign(x)*clog(sign(x)*z) + O(1/z^2)   as z -> infinity
 * The above formula works for the imaginary part as well, because
 * Im(casinh(z)) = sign(x)*atan2(sign(x)*y, fabs(x)) + O(y/z^3)
 *    as z -> infinity, uniformly in y
 */
__host__ __device__ inline
complex<double> casinh(complex<double> z)
{
  double x, y, ax, ay, rx, ry, B, sqrt_A2my2, new_y;
  int B_is_usable;
  complex<double> w;
  const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
  const double m_ln2 = 6.9314718055994531e-1; /*  0x162e42fefa39ef.0p-53 */
  x = z.real();
  y = z.imag();
  ax = fabs(x);
  ay = fabs(y);

  if (isnan(x) || isnan(y)) {
    /* casinh(+-Inf + I*NaN) = +-Inf + I*NaN */
    if (isinf(x))
      return (complex<double>(x, y + y));
    /* casinh(NaN + I*+-Inf) = opt(+-)Inf + I*NaN */
    if (isinf(y))
      return (complex<double>(y, x + x));
    /* casinh(NaN + I*0) = NaN + I*0 */
    if (y == 0)
      return (complex<double>(x + x, y));
    /*
     * All other cases involving NaN return NaN + I*NaN.
     * C99 leaves it optional whether to raise invalid if one of
     * the arguments is not NaN, so we opt not to raise it.
     */
    return (complex<double>(x + 0.0 + (y + 0.0), x + 0.0 + (y + 0.0)));
  }

  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    /* clog...() will raise inexact unless x or y is infinite. */
    if (signbit(x) == 0)
      w = clog_for_large_values(z) + m_ln2;
    else
      w = clog_for_large_values(-z) + m_ln2;
    return (complex<double>(copysign(w.real(), x), copysign(w.imag(), y)));
  }

  /* Avoid spuriously raising inexact for z = 0. */
  if (x == 0 && y == 0)
    return (z);

  /* All remaining cases are inexact. */
  raise_inexact();

  const double SQRT_6_EPSILON = 3.6500241499888571e-8; /*  0x13988e1409212e.0p-77 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (z);

  do_hard_work(ax, ay, &rx, &B_is_usable, &B, &sqrt_A2my2, &new_y);
  if (B_is_usable)
    ry = asin(B);
  else
    ry = atan2(new_y, sqrt_A2my2);
  return (complex<double>(copysign(rx, x), copysign(ry, y)));
}

/*
 * casin(z) = reverse(casinh(reverse(z)))
 * where reverse(x + I*y) = y + I*x = I*conj(z).
 */
__host__ __device__ inline
complex<double> casin(complex<double> z)
{
  complex<double> w = casinh(complex<double>(z.imag(), z.real()));

  return (complex<double>(w.imag(), w.real()));
}

/*
 * cacos(z) = PI/2 - casin(z)
 * but do the computation carefully so cacos(z) is accurate when z is
 * close to 1.
 *
 * cacos(z) = PI/2 - z + O(z^3)   as z -> 0
 *
 * cacos(z) = -sign(y)*I*clog(z) + O(1/z^2)   as z -> infinity
 * The above formula works for the real part as well, because
 * Re(cacos(z)) = atan2(fabs(y), x) + O(y/z^3)
 *    as z -> infinity, uniformly in y
 */
__host__ __device__ inline
complex<double> cacos(complex<double> z)
{
  double x, y, ax, ay, rx, ry, B, sqrt_A2mx2, new_x;
  int sx, sy;
  int B_is_usable;
  complex<double> w;
  const double pio2_hi = 1.5707963267948966e0; /*  0x1921fb54442d18.0p-52 */
  const volatile double pio2_lo = 6.1232339957367659e-17;	/*  0x11a62633145c07.0p-106 */
  const double m_ln2 = 6.9314718055994531e-1; /*  0x162e42fefa39ef.0p-53 */

  x = z.real();
  y = z.imag();
  sx = signbit(x);
  sy = signbit(y);
  ax = fabs(x);
  ay = fabs(y);

  if (isnan(x) || isnan(y)) {
    /* cacos(+-Inf + I*NaN) = NaN + I*opt(-)Inf */
    if (isinf(x))
      return (complex<double>(y + y, -infinity<double>()));
    /* cacos(NaN + I*+-Inf) = NaN + I*-+Inf */
    if (isinf(y))
      return (complex<double>(x + x, -y));
    /* cacos(0 + I*NaN) = PI/2 + I*NaN with inexact */
    if (x == 0)
      return (complex<double>(pio2_hi + pio2_lo, y + y));
    /*
     * All other cases involving NaN return NaN + I*NaN.
     * C99 leaves it optional whether to raise invalid if one of
     * the arguments is not NaN, so we opt not to raise it.
     */
    return (complex<double>(x + 0.0 + (y + 0), x + 0.0 + (y + 0)));
  }

  const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    /* clog...() will raise inexact unless x or y is infinite. */
    w = clog_for_large_values(z);
    rx = fabs(w.imag());
    ry = w.real() + m_ln2;
    if (sy == 0)
      ry = -ry;
    return (complex<double>(rx, ry));
  }

  /* Avoid spuriously raising inexact for z = 1. */
  if (x == 1.0 && y == 0.0)
    return (complex<double>(0, -y));

  /* All remaining cases are inexact. */
  raise_inexact();

  const double SQRT_6_EPSILON = 3.6500241499888571e-8; /*  0x13988e1409212e.0p-77 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (complex<double>(pio2_hi - (x - pio2_lo), -y));

  do_hard_work(ay, ax, &ry, &B_is_usable, &B, &sqrt_A2mx2, &new_x);
  if (B_is_usable) {
    if (sx == 0)
      rx = acos(B);
    else
      rx = acos(-B);
  } else {
    if (sx == 0)
      rx = atan2(sqrt_A2mx2, new_x);
    else
      rx = atan2(sqrt_A2mx2, -new_x);
  }
  if (sy == 0)
    ry = -ry;
  return (complex<double>(rx, ry));
}

/*
 * cacosh(z) = I*cacos(z) or -I*cacos(z)
 * where the sign is chosen so Re(cacosh(z)) >= 0.
 */
__host__ __device__ inline
complex<double> cacosh(complex<double> z)
{
  complex<double> w;
  double rx, ry;

  w = cacos(z);
  rx = w.real();
  ry = w.imag();
  /* cacosh(NaN + I*NaN) = NaN + I*NaN */
  if (isnan(rx) && isnan(ry))
    return (complex<double>(ry, rx));
  /* cacosh(NaN + I*+-Inf) = +Inf + I*NaN */
  /* cacosh(+-Inf + I*NaN) = +Inf + I*NaN */
  if (isnan(rx))
    return (complex<double>(fabs(ry), rx));
  /* cacosh(0 + I*NaN) = NaN + I*NaN */
  if (isnan(ry))
    return (complex<double>(ry, ry));
  return (complex<double>(fabs(ry), copysign(rx, z.imag())));
}

/*
 * Optimized version of clog() for |z| finite and larger than ~RECIP_EPSILON.
 */
__host__ __device__ inline
complex<double> clog_for_large_values(complex<double> z)
{
  double x, y;
  double ax, ay, t;
  const double m_e = 2.7182818284590452e0; /*  0x15bf0a8b145769.0p-51 */

  x = z.real();
  y = z.imag();
  ax = fabs(x);
  ay = fabs(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  /*
   * Avoid overflow in hypot() when x and y are both very large.
   * Divide x and y by E, and then add 1 to the logarithm.  This depends
   * on E being larger than sqrt(2).
   * Dividing by E causes an insignificant loss of accuracy; however
   * this method is still poor since it is uneccessarily slow.
   */
  if (ax > DBL_MAX / 2)
    return (complex<double>(log(hypot(x / m_e, y / m_e)) + 1, atan2(y, x)));

  /*
   * Avoid overflow when x or y is large.  Avoid underflow when x or
   * y is small.
   */
  const double QUARTER_SQRT_MAX = 5.966672584960165394632772e-154; /* = 0x1p509; <= sqrt(DBL_MAX) / 4 */
  const double SQRT_MIN =	1.491668146240041348658193e-154; /* = 0x1p-511; >= sqrt(DBL_MIN) */
  if (ax > QUARTER_SQRT_MAX || ay < SQRT_MIN)
    return (complex<double>(log(hypot(x, y)), atan2(y, x)));

  return (complex<double>(log(ax * ax + ay * ay) / 2, atan2(y, x)));
}

/*
 *				=================
 *				| catanh, catan |
 *				=================
 */

/*
   * sum_squares(x,y) = x*x + y*y (or just x*x if y*y would underflow).
   * Assumes x*x and y*y will not overflow.
   * Assumes x and y are finite.
   * Assumes y is non-negative.
   * Assumes fabs(x) >= DBL_EPSILON.
   */
__host__ __device__
inline double sum_squares(double x, double y)
{
  const double SQRT_MIN =	1.491668146240041348658193e-154; /* = 0x1p-511; >= sqrt(DBL_MIN) */
  /* Avoid underflow when y is small. */
  if (y < SQRT_MIN)
    return (x * x);

  return (x * x + y * y);
}

/*
 * real_part_reciprocal(x, y) = Re(1/(x+I*y)) = x/(x*x + y*y).
 * Assumes x and y are not NaN, and one of x and y is larger than
 * RECIP_EPSILON.  We avoid unwarranted underflow.  It is important to not use
 * the code creal(1/z), because the imaginary part may produce an unwanted
 * underflow.
 * This is only called in a context where inexact is always raised before
 * the call, so no effort is made to avoid or force inexact.
 */
__host__ __device__
inline double real_part_reciprocal(double x, double y)
{
  double scale;
  uint32_t hx, hy;
  int32_t ix, iy;

  /*
   * This code is inspired by the C99 document n1124.pdf, Section G.5.1,
   * example 2.
   */
  get_high_word(hx, x);
  ix = hx & 0x7ff00000;
  get_high_word(hy, y);
  iy = hy & 0x7ff00000;
  //#define	BIAS	(DBL_MAX_EXP - 1)
  const int BIAS = DBL_MAX_EXP - 1;
  /* XXX more guard digits are useful iff there is extra precision. */
  //#define	CUTOFF	(DBL_MANT_DIG / 2 + 1)	/* just half or 1 guard digit */
  const int CUTOFF = (DBL_MANT_DIG / 2 + 1);
  if (ix - iy >= CUTOFF << 20 || isinf(x))
    return (1 / x);		/* +-Inf -> +-0 is special */
  if (iy - ix >= CUTOFF << 20)
    return (x / y / y);	/* should avoid double div, but hard */
  if (ix <= (BIAS + DBL_MAX_EXP / 2 - CUTOFF) << 20)
    return (x / (x * x + y * y));
  scale = 1;
  set_high_word(scale, 0x7ff00000 - ix);	/* 2**(1-ilogb(x)) */
  x *= scale;
  y *= scale;
  return (x / (x * x + y * y) * scale);
}


/*
 * catanh(z) = log((1+z)/(1-z)) / 2
 *           = log1p(4*x / |z-1|^2) / 4
 *             + I * atan2(2*y, (1-x)*(1+x)-y*y) / 2
 *
 * catanh(z) = z + O(z^3)   as z -> 0
 *
 * catanh(z) = 1/z + sign(y)*I*PI/2 + O(1/z^3)   as z -> infinity
 * The above formula works for the real part as well, because
 * Re(catanh(z)) = x/|z|^2 + O(x/z^4)
 *    as z -> infinity, uniformly in x
 */
#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
__host__ __device__ inline
complex<double> catanh(complex<double> z)
{
  double x, y, ax, ay, rx, ry;
  const volatile double pio2_lo = 6.1232339957367659e-17; /*  0x11a62633145c07.0p-106 */
  const double pio2_hi = 1.5707963267948966e0;/*  0x1921fb54442d18.0p-52 */


  x = z.real();
  y = z.imag();
  ax = fabs(x);
  ay = fabs(y);

  /* This helps handle many cases. */
  if (y == 0 && ax <= 1)
    return (complex<double>(atanh(x), y));

  /* To ensure the same accuracy as atan(), and to filter out z = 0. */
  if (x == 0)
    return (complex<double>(x, atan(y)));

  if (isnan(x) || isnan(y)) {
    /* catanh(+-Inf + I*NaN) = +-0 + I*NaN */
    if (isinf(x))
      return (complex<double>(copysign(0.0, x), y + y));
    /* catanh(NaN + I*+-Inf) = sign(NaN)0 + I*+-PI/2 */
    if (isinf(y))
      return (complex<double>(copysign(0.0, x),
			      copysign(pio2_hi + pio2_lo, y)));
    /*
     * All other cases involving NaN return NaN + I*NaN.
     * C99 leaves it optional whether to raise invalid if one of
     * the arguments is not NaN, so we opt not to raise it.
     */
    return (complex<double>(x + 0.0 + (y + 0), x + 0.0 + (y + 0)));
  }

  const double RECIP_EPSILON = 1.0 / DBL_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON)
    return (complex<double>(real_part_reciprocal(x, y),
			    copysign(pio2_hi + pio2_lo, y)));

  const double SQRT_3_EPSILON = 2.5809568279517849e-8; /*  0x1bb67ae8584caa.0p-78 */
  if (ax < SQRT_3_EPSILON / 2 && ay < SQRT_3_EPSILON / 2) {
    /*
     * z = 0 was filtered out above.  All other cases must raise
     * inexact, but this is the only only that needs to do it
     * explicitly.
     */
    raise_inexact();
    return (z);
  }

  const double m_ln2 = 6.9314718055994531e-1; /*  0x162e42fefa39ef.0p-53 */
  if (ax == 1 && ay < DBL_EPSILON)
    rx = (m_ln2 - log(ay)) / 2;
  else
    rx = log1p(4 * ax / sum_squares(ax - 1, ay)) / 4;

  if (ax == 1)
    ry = atan2(2.0, -ay) / 2;
  else if (ay < DBL_EPSILON)
    ry = atan2(2 * ay, (1 - ax) * (1 + ax)) / 2;
  else
    ry = atan2(2 * ay, (1 - ax) * (1 + ax) - ay * ay) / 2;

  return (complex<double>(copysign(rx, x), copysign(ry, y)));
}

/*
 * catan(z) = reverse(catanh(reverse(z)))
 * where reverse(x + I*y) = y + I*x = I*conj(z).
 */
__host__ __device__ inline
complex<double>catan(complex<double> z)
{
  complex<double> w = catanh(complex<double>(z.imag(), z.real()));
  return (complex<double>(w.imag(), w.real()));
}

#endif

} // namespace complex

} // namespace detail


template <typename ValueType>
__host__ __device__
inline complex<ValueType> acos(const complex<ValueType>& z){
  const complex<ValueType> ret = thrust::asin(z);
  const ValueType pi = ValueType(3.14159265358979323846);
  return complex<ValueType>(pi/2 - ret.real(),-ret.imag());
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> asin(const complex<ValueType>& z){
  const complex<ValueType> i(0,1);
  return -i*asinh(i*z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> atan(const complex<ValueType>& z){
  const complex<ValueType> i(0,1);
  return -i*thrust::atanh(i*z);
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> acosh(const complex<ValueType>& z){
  thrust::complex<ValueType> ret((z.real() - z.imag()) * (z.real() + z.imag()) - ValueType(1.0),
				 ValueType(2.0) * z.real() * z.imag());
  ret = thrust::sqrt(ret);
  if (z.real() < ValueType(0.0)){
    ret = -ret;
  }
  ret += z;
  ret = thrust::log(ret);
  if (ret.real() < ValueType(0.0)){
    ret = -ret;
  }
  return ret;
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> asinh(const complex<ValueType>& z){
  return thrust::log(thrust::sqrt(z*z+ValueType(1))+z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> atanh(const complex<ValueType>& z){
  ValueType imag2 = z.imag() *  z.imag();
  ValueType n = ValueType(1.0) + z.real();
  n = imag2 + n * n;

  ValueType d = ValueType(1.0) - z.real();
  d = imag2 + d * d;
  complex<ValueType> ret(ValueType(0.25) * (std::log(n) - std::log(d)),0);

  d = ValueType(1.0) -  z.real() * z.real() - imag2;

  ret.imag(ValueType(0.5) * std::atan2(ValueType(2.0) * z.imag(), d));
  return ret;
}

template <>
__host__ __device__
inline complex<double> acos(const complex<double>& z){
  return detail::complex::cacos(z);
}

template <>
__host__ __device__
inline complex<double> asin(const complex<double>& z){
  return detail::complex::casin(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<double> atan(const complex<double>& z){
  return detail::complex::catan(z);
}
#endif

template <>
__host__ __device__
inline complex<double> acosh(const complex<double>& z){
  return detail::complex::cacosh(z);
}


template <>
__host__ __device__
inline complex<double> asinh(const complex<double>& z){
  return detail::complex::casinh(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<double> atanh(const complex<double>& z){
  return detail::complex::catanh(z);
}
#endif

} // namespace thrust
//thrust/detail/complex/catrig.h

//thrust/detail/complex/catrigf.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia <filipe.c.maia@gmail.com>:
 *    freebsd/lib/msun/src/catrig.c
 */

#pragma once


namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;
  
__host__ __device__ inline
      complex<float> clog_for_large_values(complex<float> z);

/*
 * The algorithm is very close to that in "Implementing the complex arcsine
 * and arccosine functions using exception handling" by T. E. Hull, Thomas F.
 * Fairgrieve, and Ping Tak Peter Tang, published in ACM Transactions on
 * Mathematical Software, Volume 23 Issue 3, 1997, Pages 299-335,
 * http://dl.acm.org/citation.cfm?id=275324.
 *
 * See catrig.c for complete comments.
 *
 * XXX comments were removed automatically, and even short ones on the right
 * of statements were removed (all of them), contrary to normal style.  Only
 * a few comments on the right of declarations remain.
 */

__host__ __device__
inline float
f(float a, float b, float hypot_a_b)
{
  if (b < 0.0f)
    return ((hypot_a_b - b) / 2.0f);
  if (b == 0.0f)
    return (a / 2.0f);
  return (a * a / (hypot_a_b + b) / 2.0f);
}

/*
 * All the hard work is contained in this function.
 * x and y are assumed positive or zero, and less than RECIP_EPSILON.
 * Upon return:
 * rx = Re(casinh(z)) = -Im(cacos(y + I*x)).
 * B_is_usable is set to 1 if the value of B is usable.
 * If B_is_usable is set to 0, sqrt_A2my2 = sqrt(A*A - y*y), and new_y = y.
 * If returning sqrt_A2my2 has potential to result in an underflow, it is
 * rescaled, and new_y is similarly rescaled.
 */
__host__ __device__ 
inline void
do_hard_work(float x, float y, float *rx, int *B_is_usable, float *B,
	     float *sqrt_A2my2, float *new_y)
{
  float R, S, A; /* A, B, R, and S are as in Hull et al. */
  float Am1, Amy; /* A-1, A-y. */
  const float A_crossover = 10; /* Hull et al suggest 1.5, but 10 works better */
  const float FOUR_SQRT_MIN = 4.336808689942017736029811e-19f;; /* =0x1p-61; >= 4 * sqrt(FLT_MIN) */
  const float B_crossover = 0.6417f; /* suggested by Hull et al */
  R = hypotf(x, y + 1);
  S = hypotf(x, y - 1);

  A = (R + S) / 2;
  if (A < 1)
    A = 1;

  if (A < A_crossover) {
    if (y == 1 && x < FLT_EPSILON * FLT_EPSILON / 128) {
      *rx = sqrtf(x);
    } else if (x >= FLT_EPSILON * fabsf(y - 1)) {
      Am1 = f(x, 1 + y, R) + f(x, 1 - y, S);
      *rx = log1pf(Am1 + sqrtf(Am1 * (A + 1)));
    } else if (y < 1) {
      *rx = x / sqrtf((1 - y) * (1 + y));
    } else {
      *rx = log1pf((y - 1) + sqrtf((y - 1) * (y + 1)));
    }
  } else {
    *rx = logf(A + sqrtf(A * A - 1));
  }

  *new_y = y;

  if (y < FOUR_SQRT_MIN) {
    *B_is_usable = 0;
    *sqrt_A2my2 = A * (2 / FLT_EPSILON);
    *new_y = y * (2 / FLT_EPSILON);
    return;
  }

  *B = y / A;
  *B_is_usable = 1;

  if (*B > B_crossover) {
    *B_is_usable = 0;
    if (y == 1 && x < FLT_EPSILON / 128) {
      *sqrt_A2my2 = sqrtf(x) * sqrtf((A + y) / 2);
    } else if (x >= FLT_EPSILON * fabsf(y - 1)) {
      Amy = f(x, y + 1, R) + f(x, y - 1, S);
      *sqrt_A2my2 = sqrtf(Amy * (A + y));
    } else if (y > 1) {
      *sqrt_A2my2 = x * (4 / FLT_EPSILON / FLT_EPSILON) * y /
	sqrtf((y + 1) * (y - 1));
      *new_y = y * (4 / FLT_EPSILON / FLT_EPSILON);
    } else {
      *sqrt_A2my2 = sqrtf((1 - y) * (1 + y));
    }
  }

}

__host__ __device__ inline
complex<float>
casinhf(complex<float> z)
{
  float x, y, ax, ay, rx, ry, B, sqrt_A2my2, new_y;
  int B_is_usable;
  complex<float> w;
  const float RECIP_EPSILON = 1.0f / FLT_EPSILON;
  const float m_ln2 = 6.9314718055994531e-1f; /*  0x162e42fefa39ef.0p-53 */
  x = z.real();
  y = z.imag();
  ax = fabsf(x);
  ay = fabsf(y);

  if (isnan(x) || isnan(y)) {
    if (isinf(x))
      return (complex<float>(x, y + y));
    if (isinf(y))
      return (complex<float>(y, x + x));
    if (y == 0)
      return (complex<float>(x + x, y));
    return (complex<float>(x + 0.0f + (y + 0), x + 0.0f + (y + 0)));
  }

  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    if (signbit(x) == 0)
      w = clog_for_large_values(z) + m_ln2;
    else
      w = clog_for_large_values(-z) + m_ln2;
    return (complex<float>(copysignf(w.real(), x),
			   copysignf(w.imag(), y)));
  }

  if (x == 0 && y == 0)
    return (z);

  raise_inexact();

  const float SQRT_6_EPSILON = 8.4572793338e-4f;	/*  0xddb3d7.0p-34 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (z);

  do_hard_work(ax, ay, &rx, &B_is_usable, &B, &sqrt_A2my2, &new_y);
  if (B_is_usable)
    ry = asinf(B);
  else
    ry = atan2f(new_y, sqrt_A2my2);
  return (complex<float>(copysignf(rx, x), copysignf(ry, y)));
}

__host__ __device__ inline
complex<float> casinf(complex<float> z)
{
  complex<float> w = casinhf(complex<float>(z.imag(), z.real()));

  return (complex<float>(w.imag(), w.real()));
}

__host__ __device__ inline
complex<float> cacosf(complex<float> z)
{
  float x, y, ax, ay, rx, ry, B, sqrt_A2mx2, new_x;
  int sx, sy;
  int B_is_usable;
  complex<float> w;
  const float pio2_hi = 1.5707963267948966e0f; /*  0x1921fb54442d18.0p-52 */
  const volatile float pio2_lo = 6.1232339957367659e-17f;	/*  0x11a62633145c07.0p-106 */
  const float m_ln2 = 6.9314718055994531e-1f; /*  0x162e42fefa39ef.0p-53 */

  x = z.real();
  y = z.imag();
  sx = signbit(x);
  sy = signbit(y);
  ax = fabsf(x);
  ay = fabsf(y);

  if (isnan(x) || isnan(y)) {
    if (isinf(x))
      return (complex<float>(y + y, -infinity<float>()));
    if (isinf(y))
      return (complex<float>(x + x, -y));
    if (x == 0)
      return (complex<float>(pio2_hi + pio2_lo, y + y));
    return (complex<float>(x + 0.0f + (y + 0), x + 0.0f + (y + 0)));
  }

  const float RECIP_EPSILON = 1.0f / FLT_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    w = clog_for_large_values(z);
    rx = fabsf(w.imag());
    ry = w.real() + m_ln2;
    if (sy == 0)
      ry = -ry;
    return (complex<float>(rx, ry));
  }

  if (x == 1 && y == 0)
    return (complex<float>(0, -y));

  raise_inexact();

  const float SQRT_6_EPSILON = 8.4572793338e-4f;	/*  0xddb3d7.0p-34 */
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4)
    return (complex<float>(pio2_hi - (x - pio2_lo), -y));

  do_hard_work(ay, ax, &ry, &B_is_usable, &B, &sqrt_A2mx2, &new_x);
  if (B_is_usable) {
    if (sx == 0)
      rx = acosf(B);
    else
      rx = acosf(-B);
  } else {
    if (sx == 0)
      rx = atan2f(sqrt_A2mx2, new_x);
    else
      rx = atan2f(sqrt_A2mx2, -new_x);
  }
  if (sy == 0)
    ry = -ry;
  return (complex<float>(rx, ry));
}

__host__ __device__ inline
complex<float> cacoshf(complex<float> z)
{
  complex<float> w;
  float rx, ry;

  w = cacosf(z);
  rx = w.real();
  ry = w.imag();
  /* cacosh(NaN + I*NaN) = NaN + I*NaN */
  if (isnan(rx) && isnan(ry))
    return (complex<float>(ry, rx));
  /* cacosh(NaN + I*+-Inf) = +Inf + I*NaN */
  /* cacosh(+-Inf + I*NaN) = +Inf + I*NaN */
  if (isnan(rx))
    return (complex<float>(fabsf(ry), rx));
  /* cacosh(0 + I*NaN) = NaN + I*NaN */
  if (isnan(ry))
    return (complex<float>(ry, ry));
  return (complex<float>(fabsf(ry), copysignf(rx, z.imag())));
}

  /*
   * Optimized version of clog() for |z| finite and larger than ~RECIP_EPSILON.
   */
__host__ __device__ inline
complex<float> clog_for_large_values(complex<float> z)
{
  float x, y;
  float ax, ay, t;
  const float m_e = 2.7182818284590452e0f; /*  0x15bf0a8b145769.0p-51 */

  x = z.real();
  y = z.imag();
  ax = fabsf(x);
  ay = fabsf(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  if (ax > FLT_MAX / 2)
    return (complex<float>(logf(hypotf(x / m_e, y / m_e)) + 1,
			   atan2f(y, x)));

  const float QUARTER_SQRT_MAX = 2.3058430092136939520000000e+18f; /* = 0x1p61; <= sqrt(FLT_MAX) / 4 */
  const float SQRT_MIN =	1.084202172485504434007453e-19f; /* 0x1p-63; >= sqrt(FLT_MIN) */
  if (ax > QUARTER_SQRT_MAX || ay < SQRT_MIN)
    return (complex<float>(logf(hypotf(x, y)), atan2f(y, x)));

  return (complex<float>(logf(ax * ax + ay * ay) / 2, atan2f(y, x)));
}

/*
 *				=================
 *				| catanh, catan |
 *				=================
 */

/*
 * sum_squares(x,y) = x*x + y*y (or just x*x if y*y would underflow).
 * Assumes x*x and y*y will not overflow.
 * Assumes x and y are finite.
 * Assumes y is non-negative.
 * Assumes fabsf(x) >= FLT_EPSILON.
 */
__host__ __device__
inline float sum_squares(float x, float y)
{
  const float SQRT_MIN =	1.084202172485504434007453e-19f; /* 0x1p-63; >= sqrt(FLT_MIN) */
  /* Avoid underflow when y is small. */
  if (y < SQRT_MIN)
    return (x * x);

  return (x * x + y * y);
}

__host__ __device__
inline float real_part_reciprocal(float x, float y)
{
  float scale;
  uint32_t hx, hy;
  int32_t ix, iy;

  get_float_word(hx, x);
  ix = hx & 0x7f800000;
  get_float_word(hy, y);
  iy = hy & 0x7f800000;
  //#define	BIAS	(FLT_MAX_EXP - 1)
  const int BIAS = FLT_MAX_EXP - 1;
  //#define	CUTOFF	(FLT_MANT_DIG / 2 + 1)
  const int CUTOFF = (FLT_MANT_DIG / 2 + 1);
  if (ix - iy >= CUTOFF << 23 || isinf(x))
    return (1 / x);
  if (iy - ix >= CUTOFF << 23)
    return (x / y / y);
  if (ix <= (BIAS + FLT_MAX_EXP / 2 - CUTOFF) << 23)
    return (x / (x * x + y * y));
  set_float_word(scale, 0x7f800000 - ix);
  x *= scale;
  y *= scale;
  return (x / (x * x + y * y) * scale);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
__host__ __device__ inline
complex<float> catanhf(complex<float> z)
{
  float x, y, ax, ay, rx, ry;
  const volatile float pio2_lo = 6.1232339957367659e-17f; /*  0x11a62633145c07.0p-106 */
  const float pio2_hi = 1.5707963267948966e0f;/*  0x1921fb54442d18.0p-52 */


  x = z.real();
  y = z.imag();
  ax = fabsf(x);
  ay = fabsf(y);


  if (y == 0 && ax <= 1)
    return (complex<float>(atanhf(x), y));

  if (x == 0)
    return (complex<float>(x, atanf(y)));

  if (isnan(x) || isnan(y)) {
    if (isinf(x))
      return (complex<float>(copysignf(0, x), y + y));
    if (isinf(y))
      return (complex<float>(copysignf(0, x),
			     copysignf(pio2_hi + pio2_lo, y)));
    return (complex<float>(x + 0.0f + (y + 0.0f), x + 0.0f + (y + 0.0f)));
  }

  const float RECIP_EPSILON = 1.0f / FLT_EPSILON;
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON)
    return (complex<float>(real_part_reciprocal(x, y),
			   copysignf(pio2_hi + pio2_lo, y)));

  const float SQRT_3_EPSILON = 5.9801995673e-4f; /*  0x9cc471.0p-34 */
  if (ax < SQRT_3_EPSILON / 2 && ay < SQRT_3_EPSILON / 2) {
    raise_inexact();
    return (z);
  }

  const float m_ln2 = 6.9314718056e-1f; /*  0xb17218.0p-24 */
  if (ax == 1 && ay < FLT_EPSILON)
    rx = (m_ln2 - logf(ay)) / 2;
  else
    rx = log1pf(4 * ax / sum_squares(ax - 1, ay)) / 4;

  if (ax == 1)
    ry = atan2f(2, -ay) / 2;
  else if (ay < FLT_EPSILON)
    ry = atan2f(2 * ay, (1 - ax) * (1 + ax)) / 2;
  else
    ry = atan2f(2 * ay, (1 - ax) * (1 + ax) - ay * ay) / 2;

  return (complex<float>(copysignf(rx, x), copysignf(ry, y)));
}

__host__ __device__ inline
complex<float>catanf(complex<float> z){
  complex<float> w = catanhf(complex<float>(z.imag(), z.real()));
  return (complex<float>(w.imag(), w.real()));
}
#endif

} // namespace complex

} // namespace detail


template <>
__host__ __device__
inline complex<float> acos(const complex<float>& z){
  return detail::complex::cacosf(z);
}

template <>
__host__ __device__
inline complex<float> asin(const complex<float>& z){
  return detail::complex::casinf(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<float> atan(const complex<float>& z){
  return detail::complex::catanf(z);
}
#endif

template <>
__host__ __device__
inline complex<float> acosh(const complex<float>& z){
  return detail::complex::cacoshf(z);
}


template <>
__host__ __device__
inline complex<float> asinh(const complex<float>& z){
  return detail::complex::casinhf(z);
}

#if THRUST_CPP_DIALECT >= 2011 || THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
template <>
__host__ __device__
inline complex<float> atanh(const complex<float>& z){
  return detail::complex::catanhf(z);
}
#endif

} // namespace thrust
//thrust/detail/complex/catrigf.h
//thrust/detail/complex/cexpf.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2011 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/* adapted from FreeBSD:
 *    lib/msun/src/s_cexpf.c
 *    lib/msun/src/k_exp.c
 *
 */

#pragma once

namespace thrust{
using std::complex;

namespace detail{
namespace complex{

__host__ __device__ inline
float frexp_expf(float x, int *expt){
  const uint32_t k = 235;                 /* constant for reduction */
  const float kln2 =  162.88958740F;       /* k * ln2 */
	
  // should this be a double instead?
  float exp_x;
  uint32_t hx;
	
  exp_x = expf(x - kln2);
  get_float_word(hx, exp_x);
  *expt = (hx >> 23) - (0x7f + 127) + k;
  set_float_word(exp_x, (hx & 0x7fffff) | ((0x7f + 127) << 23));
  return (exp_x);
}
      
__host__ __device__ inline
complex<float> 
ldexp_cexpf(complex<float> z, int expt)
{
  float x, y, exp_x, scale1, scale2;
  int ex_expt, half_expt;
	
  x = z.real();
  y = z.imag();
  exp_x = frexp_expf(x, &ex_expt);
  expt += ex_expt;
	
  half_expt = expt / 2;
  set_float_word(scale1, (0x7f + half_expt) << 23);
  half_expt = expt - half_expt;
  set_float_word(scale2, (0x7f + half_expt) << 23);
	
  return (complex<float>(std::cos(y) * exp_x * scale1 * scale2,
			 std::sin(y) * exp_x * scale1 * scale2));
}
      
__host__ __device__ inline
complex<float> cexpf(const complex<float>& z){
  float x, y, exp_x;
  uint32_t hx, hy;

  const uint32_t
    exp_ovfl  = 0x42b17218,		/* MAX_EXP * ln2 ~= 88.722839355 */
    cexp_ovfl = 0x43400074;		/* (MAX_EXP - MIN_DENORM_EXP) * ln2 */

  x = z.real();
  y = z.imag();

  get_float_word(hy, y);
  hy &= 0x7fffffff;

  /* cexp(x + I 0) = exp(x) + I 0 */
  if (hy == 0)
    return (complex<float>(std::exp(x), y));
  get_float_word(hx, x);
  /* cexp(0 + I y) = cos(y) + I sin(y) */
  if ((hx & 0x7fffffff) == 0){
    return (complex<float>(std::cos(y), std::sin(y)));
  }
  if (hy >= 0x7f800000) {
    if ((hx & 0x7fffffff) != 0x7f800000) {
      /* cexp(finite|NaN +- I Inf|NaN) = NaN + I NaN */
      return (complex<float>(y - y, y - y));
    } else if (hx & 0x80000000) {
      /* cexp(-Inf +- I Inf|NaN) = 0 + I 0 */
      return (complex<float>(0.0, 0.0));
    } else {
      /* cexp(+Inf +- I Inf|NaN) = Inf + I NaN */
      return (complex<float>(x, y - y));
    }
  }

  if (hx >= exp_ovfl && hx <= cexp_ovfl) {
    /*
     * x is between 88.7 and 192, so we must scale to avoid
     * overflow in expf(x).
     */
    return (ldexp_cexpf(z, 0));
  } else {
    /*
     * Cases covered here:
     *  -  x < exp_ovfl and exp(x) won't overflow (common case)
     *  -  x > cexp_ovfl, so exp(x) * s overflows for all s > 0
     *  -  x = +-Inf (generated by exp())
     *  -  x = NaN (spurious inexact exception from y)
     */
    exp_x = std::exp(x);
    return (complex<float>(exp_x * std::cos(y), exp_x * std::sin(y)));
  }
}

} // namespace complex

} // namespace detail

__host__ __device__
inline complex<float> exp(const complex<float>& z){    
  return detail::complex::cexpf(z);
}    
  
} // namespace thrust
//thrust/detail/complex/cexpf.h
//thrust/detail/complex/cexp.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2011 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/* adapted from FreeBSD:
 *    lib/msun/src/s_cexp.c
 *    lib/msun/src/k_exp.c
 *
 */

#pragma once

namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	
/*
 * Compute exp(x), scaled to avoid spurious overflow.  An exponent is
 * returned separately in 'expt'.
 *
 * Input:  ln(DBL_MAX) <= x < ln(2 * DBL_MAX / DBL_MIN_DENORM) ~= 1454.91
 * Output: 2**1023 <= y < 2**1024
 */
__host__ __device__ inline
	double frexp_exp(double x, int *expt){
  const uint32_t k = 1799;		/* constant for reduction */
  const double kln2 =  1246.97177782734161156;	/* k * ln2 */
	
  double exp_x;
  uint32_t hx;
	
  /*
   * We use exp(x) = exp(x - kln2) * 2**k, carefully chosen to
   * minimize |exp(kln2) - 2**k|.  We also scale the exponent of
   * exp_x to MAX_EXP so that the result can be multiplied by
   * a tiny number without losing accuracy due to denormalization.
   */
  exp_x = exp(x - kln2);
  get_high_word(hx, exp_x);
  *expt = (hx >> 20) - (0x3ff + 1023) + k;
  set_high_word(exp_x, (hx & 0xfffff) | ((0x3ff + 1023) << 20));
  return (exp_x);
}
      
      
__host__ __device__ inline
complex<double>	ldexp_cexp(complex<double> z, int expt){
  double x, y, exp_x, scale1, scale2;
  int ex_expt, half_expt;
	
  x = z.real();
  y = z.imag();
  exp_x = frexp_exp(x, &ex_expt);
  expt += ex_expt;
	
  /*
   * Arrange so that scale1 * scale2 == 2**expt.  We use this to
   * compensate for scalbn being horrendously slow.
   */
  half_expt = expt / 2;
  insert_words(scale1, (0x3ff + half_expt) << 20, 0);
  half_expt = expt - half_expt;
  insert_words(scale2, (0x3ff + half_expt) << 20, 0);
	
  return (complex<double>(cos(y) * exp_x * scale1 * scale2,
			  sin(y) * exp_x * scale1 * scale2));
}
	

__host__ __device__ inline
complex<double> cexp(const complex<double>& z){
  double x, y, exp_x;
  uint32_t hx, hy, lx, ly;

  const uint32_t
    exp_ovfl  = 0x40862e42,			/* high bits of MAX_EXP * ln2 ~= 710 */
    cexp_ovfl = 0x4096b8e4;			/* (MAX_EXP - MIN_DENORM_EXP) * ln2 */

	  
  x = z.real();
  y = z.imag();
	  
  extract_words(hy, ly, y);
  hy &= 0x7fffffff;
	  
  /* cexp(x + I 0) = exp(x) + I 0 */
  if ((hy | ly) == 0)
    return (complex<double>(exp(x), y));
  extract_words(hx, lx, x);
  /* cexp(0 + I y) = cos(y) + I sin(y) */
  if (((hx & 0x7fffffff) | lx) == 0)
    return (complex<double>(cos(y), sin(y)));
	  
  if (hy >= 0x7ff00000) {
    if (lx != 0 || (hx & 0x7fffffff) != 0x7ff00000) {
      /* cexp(finite|NaN +- I Inf|NaN) = NaN + I NaN */
      return (complex<double>(y - y, y - y));
    } else if (hx & 0x80000000) {
      /* cexp(-Inf +- I Inf|NaN) = 0 + I 0 */
      return (complex<double>(0.0, 0.0));
    } else {
      /* cexp(+Inf +- I Inf|NaN) = Inf + I NaN */
      return (complex<double>(x, y - y));
    }
  }
	  
  if (hx >= exp_ovfl && hx <= cexp_ovfl) {
    /*
     * x is between 709.7 and 1454.3, so we must scale to avoid
     * overflow in exp(x).
     */
    return (ldexp_cexp(z, 0));
  } else {
    /*
     * Cases covered here:
     *  -  x < exp_ovfl and exp(x) won't overflow (common case)
     *  -  x > cexp_ovfl, so exp(x) * s overflows for all s > 0
     *  -  x = +-Inf (generated by exp())
     *  -  x = NaN (spurious inexact exception from y)
     */
    exp_x = std::exp(x);
    return (complex<double>(exp_x * cos(y), exp_x * sin(y)));
  }
}
	
} // namespace complex
 
} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> exp(const complex<ValueType>& z){    
  return polar(std::exp(z.real()),z.imag());
}

template <>
__host__ __device__
inline complex<double> exp(const complex<double>& z){    
  return detail::complex::cexp(z);
}

} // namespace thrust
//thrust/detail/complex/cexp.h
//thrust/detail/complex/ccoshf.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2005 Bruce D. Evans and Steven G. Kargl
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* adapted from FreeBSD:
 *    lib/msun/src/s_ccoshf.c
 */


#pragma once


namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;
      
__host__ __device__ inline
complex<float> ccoshf(const complex<float>& z){
  float x, y, h;
  uint32_t hx, hy, ix, iy;
  const float huge = 1.70141183460469231731687303716e+38; //0x1p127;	
  
  
  x = z.real();
  y = z.imag();
  
  get_float_word(hx, x);
  get_float_word(hy, y);
  
  ix = 0x7fffffff & hx;
  iy = 0x7fffffff & hy;
  if (ix < 0x7f800000 && iy < 0x7f800000) {
    if (iy == 0){
      return (complex<float>(coshf(x), x * y));
    }
    if (ix < 0x41100000){	/* small x: normal case */
      return (complex<float>(coshf(x) * cosf(y), sinhf(x) * sinf(y)));
    }
    /* |x| >= 9, so cosh(x) ~= exp(|x|) */
    if (ix < 0x42b17218) {
      /* x < 88.7: expf(|x|) won't overflow */
      h = expf(fabsf(x)) * 0.5f;
      return (complex<float>(h * cosf(y), copysignf(h, x) * sinf(y)));
    } else if (ix < 0x4340b1e7) {
      /* x < 192.7: scale to avoid overflow */
      thrust::complex<float> z_;
      z_ = ldexp_cexpf(complex<float>(fabsf(x), y), -1);
      return (complex<float>(z_.real(), z_.imag() * copysignf(1.0f, x)));
    } else {
      /* x >= 192.7: the result always overflows */
      h = huge * x;
      return (complex<float>(h * h * cosf(y), h * sinf(y)));
    }
  }
  
  if (ix == 0 && iy >= 0x7f800000){
    return (complex<float>(y - y, copysignf(0.0f, x * (y - y))));
  }
  if (iy == 0 && ix >= 0x7f800000) {
    if ((hx & 0x7fffff) == 0)
      return (complex<float>(x * x, copysignf(0.0f, x) * y));
    return (complex<float>(x * x, copysignf(0.0f, (x + x) * y)));
  }
  
  if (ix < 0x7f800000 && iy >= 0x7f800000){
    return (complex<float>(y - y, x * (y - y)));
  }
  
  if (ix >= 0x7f800000 && (hx & 0x7fffff) == 0) {
    if (iy >= 0x7f800000)
      return (complex<float>(x * x, x * (y - y)));
    return (complex<float>((x * x) * cosf(y), x * sinf(y)));
  }
  return (complex<float>((x * x) * (y - y), (x + x) * (y - y)));
}
  
__host__ __device__ inline
complex<float> ccosf(const complex<float>& z){	
  return (ccoshf(complex<float>(-z.imag(), z.real())));
}

} // namespace complex

} // namespace detail

__host__ __device__
inline complex<float> cos(const complex<float>& z){
  return detail::complex::ccosf(z);
}
  
__host__ __device__
inline complex<float> cosh(const complex<float>& z){
  return detail::complex::ccoshf(z);
}
  
} // namespace thrust
//thrust/detail/complex/ccoshf.h
//thrust/detail/complex/ccosh.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2005 Bruce D. Evans and Steven G. Kargl
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* adapted from FreeBSD:
 *    lib/msun/src/s_ccosh.c
 */

#pragma once

namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

/*
 * Hyperbolic cosine of a complex argument z = x + i y.
 *
 * cosh(z) = cosh(x+iy)
 *         = cosh(x) cos(y) + i sinh(x) sin(y).
 *
 * Exceptional values are noted in the comments within the source code.
 * These values and the return value were taken from n1124.pdf.
 */
      
__host__ __device__ inline
thrust::complex<double> ccosh(const thrust::complex<double>& z){
  

  const double huge = 8.98846567431157953864652595395e+307; // 0x1p1023
  double x, y, h;
  uint32_t hx, hy, ix, iy, lx, ly;

  x = z.real();
  y = z.imag();

  extract_words(hx, lx, x);
  extract_words(hy, ly, y);

  ix = 0x7fffffff & hx;
  iy = 0x7fffffff & hy;

  /* Handle the nearly-non-exceptional cases where x and y are finite. */
  if (ix < 0x7ff00000 && iy < 0x7ff00000) {
    if ((iy | ly) == 0)
      return (thrust::complex<double>(::cosh(x), x * y));
    if (ix < 0x40360000)	/* small x: normal case */
      return (thrust::complex<double>(::cosh(x) * ::cos(y), ::sinh(x) * ::sin(y)));

    /* |x| >= 22, so cosh(x) ~= exp(|x|) */
    if (ix < 0x40862e42) {
      /* x < 710: exp(|x|) won't overflow */
      h = ::exp(::fabs(x)) * 0.5;
      return (thrust::complex<double>(h * cos(y), copysign(h, x) * sin(y)));
    } else if (ix < 0x4096bbaa) {
      /* x < 1455: scale to avoid overflow */
      thrust::complex<double> z_;
      z_ = ldexp_cexp(thrust::complex<double>(fabs(x), y), -1);
      return (thrust::complex<double>(z_.real(), z_.imag() * copysign(1.0, x)));
    } else {
      /* x >= 1455: the result always overflows */
      h = huge * x;
      return (thrust::complex<double>(h * h * cos(y), h * sin(y)));
    }
  }

  /*
   * cosh(+-0 +- I Inf) = dNaN + I sign(d(+-0, dNaN))0.
   * The sign of 0 in the result is unspecified.  Choice = normally
   * the same as dNaN.  Raise the invalid floating-point exception.
   *
   * cosh(+-0 +- I NaN) = d(NaN) + I sign(d(+-0, NaN))0.
   * The sign of 0 in the result is unspecified.  Choice = normally
   * the same as d(NaN).
   */
  if ((ix | lx) == 0 && iy >= 0x7ff00000)
    return (thrust::complex<double>(y - y, copysign(0.0, x * (y - y))));

  /*
   * cosh(+-Inf +- I 0) = +Inf + I (+-)(+-)0.
   *
   * cosh(NaN +- I 0)   = d(NaN) + I sign(d(NaN, +-0))0.
   * The sign of 0 in the result is unspecified.
   */
  if ((iy | ly) == 0 && ix >= 0x7ff00000) {
    if (((hx & 0xfffff) | lx) == 0)
      return (thrust::complex<double>(x * x, copysign(0.0, x) * y));
    return (thrust::complex<double>(x * x, copysign(0.0, (x + x) * y)));
  }

  /*
   * cosh(x +- I Inf) = dNaN + I dNaN.
   * Raise the invalid floating-point exception for finite nonzero x.
   *
   * cosh(x + I NaN) = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception for finite
   * nonzero x.  Choice = don't raise (except for signaling NaNs).
   */
  if (ix < 0x7ff00000 && iy >= 0x7ff00000)
    return (thrust::complex<double>(y - y, x * (y - y)));

  /*
   * cosh(+-Inf + I NaN)  = +Inf + I d(NaN).
   *
   * cosh(+-Inf +- I Inf) = +Inf + I dNaN.
   * The sign of Inf in the result is unspecified.  Choice = always +.
   * Raise the invalid floating-point exception.
   *
   * cosh(+-Inf + I y)   = +Inf cos(y) +- I Inf sin(y)
   */
  if (ix >= 0x7ff00000 && ((hx & 0xfffff) | lx) == 0) {
    if (iy >= 0x7ff00000)
      return (thrust::complex<double>(x * x, x * (y - y)));
    return (thrust::complex<double>((x * x) * cos(y), x * sin(y)));
  }

  /*
   * cosh(NaN + I NaN)  = d(NaN) + I d(NaN).
   *
   * cosh(NaN +- I Inf) = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception.
   * Choice = raise.
   *
   * cosh(NaN + I y)    = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception for finite
   * nonzero y.  Choice = don't raise (except for signaling NaNs).
   */
  return (thrust::complex<double>((x * x) * (y - y), (x + x) * (y - y)));
}


__host__ __device__ inline
thrust::complex<double> ccos(const thrust::complex<double>& z){	
  /* ccos(z) = ccosh(I * z) */
  return (ccosh(thrust::complex<double>(-z.imag(), z.real())));
}

} // namespace complex

} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> cos(const complex<ValueType>& z){
  const ValueType re = z.real();
  const ValueType im = z.imag();
  return complex<ValueType>(std::cos(re) * std::cosh(im), 
			    -std::sin(re) * std::sinh(im));
}
  
template <typename ValueType>
__host__ __device__
inline complex<ValueType> cosh(const complex<ValueType>& z){
  const ValueType re = z.real();
  const ValueType im = z.imag();
  return complex<ValueType>(std::cosh(re) * std::cos(im), 
			    std::sinh(re) * std::sin(im));
}

template <>
__host__ __device__
inline thrust::complex<double> cos(const thrust::complex<double>& z){
  return detail::complex::ccos(z);
}

template <>
__host__ __device__
inline thrust::complex<double> cosh(const thrust::complex<double>& z){
  return detail::complex::ccosh(z);
}

} // namespace thrust
//thrust/detail/complex/ccosh.h

//thrust/detail/complex/clogf.h
/*
 *  Copyright 2008-2021 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/* adapted from FreeBSDs msun:*/

#pragma once


namespace thrust{
using std::complex;
namespace detail{
namespace complex{

using std::complex;

/* round down to 8 = 24/3 bits */
__host__ __device__ inline
float trim(float x){
  uint32_t hx;
  get_float_word(hx, x);
  hx &= 0xffff0000;
  float ret;
  set_float_word(ret,hx);
  return ret;
}


__host__ __device__ inline
complex<float> clogf(const complex<float>& z){

  // Adapted from FreeBSDs msun
  float x, y;
  float ax, ay;
  float x0, y0, x1, y1, x2, y2, t, hm1;
  float val[12];
  int i, sorted;
  const float e = 2.7182818284590452354f;

  x = z.real();
  y = z.imag();

  /* Handle NaNs using the general formula to mix them right. */
  if (x != x || y != y){
    return (complex<float>(std::log(norm(z)), std::atan2(y, x)));
  }

  ax = std::abs(x);
  ay = std::abs(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  /*
   * To avoid unnecessary overflow, if x and y are very large, divide x
   * and y by M_E, and then add 1 to the logarithm.  This depends on
   * M_E being larger than sqrt(2).
   * There is a potential loss of accuracy caused by dividing by M_E,
   * but this case should happen extremely rarely.
   */
  // For high values of ay -> hypotf(FLT_MAX,ay) = inf
  // We expect that for values at or below ay = 1e34f this should not happen
  if (ay > 1e34f){
    return (complex<float>(std::log(hypotf(x / e, y / e)) + 1.0f, std::atan2(y, x)));
  }
  if (ax == 1.f) {
    if (ay < 1e-19f){
      return (complex<float>((ay * 0.5f) * ay, std::atan2(y, x)));
    }
    return (complex<float>(log1pf(ay * ay) * 0.5f, std::atan2(y, x)));
  }

  /*
   * Because atan2 and hypot conform to C99, this also covers all the
   * edge cases when x or y are 0 or infinite.
   */
  if (ax < 1e-6f || ay < 1e-6f || ax > 1e6f || ay > 1e6f){
    return (complex<float>(std::log(hypotf(x, y)), std::atan2(y, x)));
  }

  /*
   * From this point on, we don't need to worry about underflow or
   * overflow in calculating ax*ax or ay*ay.
   */

  /* Some easy cases. */

  if (ax >= 1.0f){
    return (complex<float>(log1pf((ax-1.f)*(ax+1.f) + ay*ay) * 0.5f, atan2(y, x)));
  }

  if (ax*ax + ay*ay <= 0.7f){
    return (complex<float>(std::log(ax*ax + ay*ay) * 0.5f, std::atan2(y, x)));
  }

  /*
   * Take extra care so that ULP of real part is small if hypot(x,y) is
   * moderately close to 1.
   */


  x0 = trim(ax);
  ax = ax-x0;
  x1 = trim(ax);
  x2 = ax-x1;
  y0 = trim(ay);
  ay = ay-y0;
  y1 = trim(ay);
  y2 = ay-y1;

  val[0] = x0*x0;
  val[1] = y0*y0;
  val[2] = 2*x0*x1;
  val[3] = 2*y0*y1;
  val[4] = x1*x1;
  val[5] = y1*y1;
  val[6] = 2*x0*x2;
  val[7] = 2*y0*y2;
  val[8] = 2*x1*x2;
  val[9] = 2*y1*y2;
  val[10] = x2*x2;
  val[11] = y2*y2;

  /* Bubble sort. */

  do {
    sorted = 1;
    for (i=0;i<11;i++) {
      if (val[i] < val[i+1]) {
	sorted = 0;
	t = val[i];
	val[i] = val[i+1];
	val[i+1] = t;
      }
    }
  } while (!sorted);

  hm1 = -1;
  for (i=0;i<12;i++){
    hm1 += val[i];
  }
  return (complex<float>(0.5f * log1pf(hm1), atan2(y, x)));
}

} // namespace complex

} // namespace detail

__host__ __device__
inline complex<float> log(const complex<float>& z){
  return detail::complex::clogf(z);
}

} // namespace thrust

//thrust/detail/complex/clogf.h
//thrust/detail/complex/clog.h
/*
 *  Copyright 2008-2021 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/* adapted from FreeBSDs msun:*/


#pragma once



namespace thrust{
using std::complex;
namespace detail{
namespace complex{

using std::complex;

/* round down to 18 = 54/3 bits */
__host__ __device__ inline
double trim(double x){
  uint32_t hi;
  get_high_word(hi, x);
  insert_words(x, hi &0xfffffff8, 0);
  return x;
}


__host__ __device__ inline
complex<double> clog(const complex<double>& z){

  // Adapted from FreeBSDs msun
  double x, y;
  double ax, ay;
  double x0, y0, x1, y1, x2, y2, t, hm1;
  double val[12];
  int i, sorted;
  const double e = 2.7182818284590452354;

  x = z.real();
  y = z.imag();

  /* Handle NaNs using the general formula to mix them right. */
  if (x != x || y != y){
    return (complex<double>(std::log(norm(z)), std::atan2(y, x)));
  }

  ax = std::abs(x);
  ay = std::abs(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  /*
   * To avoid unnecessary overflow, if x and y are very large, divide x
   * and y by M_E, and then add 1 to the logarithm.  This depends on
   * M_E being larger than sqrt(2).
   * There is a potential loss of accuracy caused by dividing by M_E,
   * but this case should happen extremely rarely.
   */
  //    if (ay > 5e307){
  // For high values of ay -> hypotf(DBL_MAX,ay) = inf
  // We expect that for values at or below ay = 5e307 this should not happen
  if (ay > 5e307){
    return (complex<double>(std::log(hypot(x / e, y / e)) + 1.0, std::atan2(y, x)));
  }
  if (ax == 1.) {
    if (ay < 1e-150){
      return (complex<double>((ay * 0.5) * ay, std::atan2(y, x)));
    }
    return (complex<double>(log1p(ay * ay) * 0.5, std::atan2(y, x)));
  }

  /*
   * Because atan2 and hypot conform to C99, this also covers all the
   * edge cases when x or y are 0 or infinite.
   */
  if (ax < 1e-50 || ay < 1e-50 || ax > 1e50 || ay > 1e50){
    return (complex<double>(std::log(hypot(x, y)), std::atan2(y, x)));
  }

  /*
   * From this point on, we don't need to worry about underflow or
   * overflow in calculating ax*ax or ay*ay.
   */

  /* Some easy cases. */

  if (ax >= 1.0){
    return (complex<double>(log1p((ax-1)*(ax+1) + ay*ay) * 0.5, atan2(y, x)));
  }

  if (ax*ax + ay*ay <= 0.7){
    return (complex<double>(std::log(ax*ax + ay*ay) * 0.5, std::atan2(y, x)));
  }

  /*
   * Take extra care so that ULP of real part is small if hypot(x,y) is
   * moderately close to 1.
   */


  x0 = trim(ax);
  ax = ax-x0;
  x1 = trim(ax);
  x2 = ax-x1;
  y0 = trim(ay);
  ay = ay-y0;
  y1 = trim(ay);
  y2 = ay-y1;

  val[0] = x0*x0;
  val[1] = y0*y0;
  val[2] = 2*x0*x1;
  val[3] = 2*y0*y1;
  val[4] = x1*x1;
  val[5] = y1*y1;
  val[6] = 2*x0*x2;
  val[7] = 2*y0*y2;
  val[8] = 2*x1*x2;
  val[9] = 2*y1*y2;
  val[10] = x2*x2;
  val[11] = y2*y2;

  /* Bubble sort. */

  do {
    sorted = 1;
    for (i=0;i<11;i++) {
      if (val[i] < val[i+1]) {
	sorted = 0;
	t = val[i];
	val[i] = val[i+1];
	val[i+1] = t;
      }
    }
  } while (!sorted);

  hm1 = -1;
  for (i=0;i<12;i++){
    hm1 += val[i];
  }
  return (complex<double>(0.5 * log1p(hm1), atan2(y, x)));
}

} // namespace complex

} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> log(const complex<ValueType>& z){
  return complex<ValueType>(std::log(thrust::abs(z)),thrust::arg(z));
}

template <>
__host__ __device__
inline complex<double> log(const complex<double>& z){
  return detail::complex::clog(z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> log10(const complex<ValueType>& z){
  // Using the explicit literal prevents compile time warnings in
  // devices that don't support doubles
  return thrust::log(z)/ValueType(2.30258509299404568402);
}

} // namespace thrust

//thrust/detail/complex/clog.h
// //thrust/detail/complex/cpow.h
// /*
//  *  Copyright 2008-2013 NVIDIA Corporation
//  *  Copyright 2013 Filipe RNC Maia
//  *
//  *  Licensed under the Apache License, Version 2.0 (the "License");
//  *  you may not use this file except in compliance with the License.
//  *  You may obtain a copy of the License at
//  *
//  *      http://www.apache.org/licenses/LICENSE-2.0
//  *
//  *  Unless required by applicable law or agreed to in writing, software
//  *  distributed under the License is distributed on an "AS IS" BASIS,
//  *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  *  See the License for the specific language governing permissions and
//  *  limitations under the License.
//  */

// #pragma once



// namespace thrust{
// using std::complex;

// template <typename T0, typename T1>
// __host__ __device__
// complex<typename detail::promoted_numerical_type<T0, T1>::type>
// pow(const complex<T0>& x, const complex<T1>& y)
// {
//   typedef typename detail::promoted_numerical_type<T0, T1>::type T;
//   return exp(log(complex<T>(x)) * complex<T>(y));
// }

// template <typename T0, typename T1>
// __host__ __device__
// complex<typename detail::promoted_numerical_type<T0, T1>::type>
// pow(const complex<T0>& x, const T1& y)
// {
//   typedef typename detail::promoted_numerical_type<T0, T1>::type T;
//   return exp(log(complex<T>(x)) * T(y));
// }

// template <typename T0, typename T1>
// __host__ __device__
// complex<typename detail::promoted_numerical_type<T0, T1>::type>
// pow(const T0& x, const complex<T1>& y)
// {
//   typedef typename detail::promoted_numerical_type<T0, T1>::type T;
//   // Find `log` by ADL.
//   using std::log;
//   return exp(log(T(x)) * complex<T>(y));
// }

// } // namespace thrust

// //thrust/detail/complex/cpow.h
//thrust/detail/complex/cproj.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once


namespace thrust{
using std::complex;
namespace detail{
namespace complex{	 
__host__ __device__
inline complex<float> cprojf(const complex<float>& z){
  if(!isinf(z.real()) && !isinf(z.imag())){
    return z;
  }else{
    // std::numeric_limits<T>::infinity() doesn't run on the GPU
    return complex<float>(infinity<float>(), copysignf(0.0, z.imag()));
  }
}
  
__host__ __device__
inline complex<double> cproj(const complex<double>& z){
  if(!isinf(z.real()) && !isinf(z.imag())){
    return z;
  }else{
    // std::numeric_limits<T>::infinity() doesn't run on the GPU
    return complex<double>(infinity<double>(), copysign(0.0, z.imag()));
  }
}

}
 
}

template <typename T>
__host__ __device__
inline thrust::complex<T> proj(const thrust::complex<T>& z){
  return detail::complex::cproj(z);
}
  

template <>
__host__ __device__
inline thrust::complex<double> proj(const thrust::complex<double>& z){
  return detail::complex::cproj(z);
}
  
template <>
__host__ __device__
inline thrust::complex<float> proj(const thrust::complex<float>& z){
  return detail::complex::cprojf(z);
}

} // namespace thrust
//thrust/detail/complex/cproj.h
//thrust/detail/complex/csinhf.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2005 Bruce D. Evans and Steven G. Kargl
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* adapted from FreeBSD:
 *    lib/msun/src/s_csinhf.c
 */


#pragma once



namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;

__host__ __device__ inline
complex<float> csinhf(const complex<float>& z){

  float x, y, h;
  uint32_t hx, hy, ix, iy;

  const float huge = 1.70141183460469231731687303716e+38; //0x1p127;

  x = z.real();
  y = z.imag();

  get_float_word(hx, x);
  get_float_word(hy, y);

  ix = 0x7fffffff & hx;
  iy = 0x7fffffff & hy;

  if (ix < 0x7f800000 && iy < 0x7f800000) {
    if (iy == 0)
      return (complex<float>(sinhf(x), y));
    if (ix < 0x41100000)	/* small x: normal case */
      return (complex<float>(sinhf(x) * cosf(y), coshf(x) * sinf(y)));

    /* |x| >= 9, so cosh(x) ~= exp(|x|) */
    if (ix < 0x42b17218) {
      /* x < 88.7: expf(|x|) won't overflow */
      h = expf(fabsf(x)) * 0.5f;
      return (complex<float>(copysignf(h, x) * cosf(y), h * sinf(y)));
    } else if (ix < 0x4340b1e7) {
      /* x < 192.7: scale to avoid overflow */
      complex<float> z_ = ldexp_cexpf(complex<float>(fabsf(x), y), -1);
      return (complex<float>(z_.real() * copysignf(1.0f, x), z_.imag()));
    } else {
      /* x >= 192.7: the result always overflows */
      h = huge * x;
      return (complex<float>(h * cosf(y), h * h * sinf(y)));
    }
  }

  if (ix == 0 && iy >= 0x7f800000)
    return (complex<float>(copysignf(0, x * (y - y)), y - y));

  if (iy == 0 && ix >= 0x7f800000) {
    if ((hx & 0x7fffff) == 0)
      return (complex<float>(x, y));
    return (complex<float>(x, copysignf(0.0f, y)));
  }

  if (ix < 0x7f800000 && iy >= 0x7f800000)
    return (complex<float>(y - y, x * (y - y)));

  if (ix >= 0x7f800000 && (hx & 0x7fffff) == 0) {
    if (iy >= 0x7f800000)
      return (complex<float>(x * x, x * (y - y)));
    return (complex<float>(x * cosf(y), infinity<float>() * sinf(y)));
  }

  return (complex<float>((x * x) * (y - y), (x + x) * (y - y)));
}

__host__ __device__ inline
complex<float> csinf(complex<float> z){
  z = csinhf(complex<float>(-z.imag(), z.real()));
  return (complex<float>(z.imag(), -z.real()));
}
      
} // namespace complex

} // namespace detail
  
__host__ __device__
inline complex<float> sin(const complex<float>& z){
  return detail::complex::csinf(z);
}

__host__ __device__
inline complex<float> sinh(const complex<float>& z){
  return detail::complex::csinhf(z);
}

} // namespace thrust
//thrust/detail/complex/csinhf.h
//thrust/detail/complex/csinh.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2005 Bruce D. Evans and Steven G. Kargl
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* adapted from FreeBSD:
 *    lib/msun/src/s_csinh.c
 */


#pragma once



namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;

__host__ __device__ inline
complex<double> csinh(const complex<double>& z){
  double x, y, h;
  uint32_t hx, hy, ix, iy, lx, ly;
  const double huge = 8.98846567431157953864652595395e+307; // 0x1p1023;

  x = z.real();
  y = z.imag();

  extract_words(hx, lx, x);
  extract_words(hy, ly, y);

  ix = 0x7fffffff & hx;
  iy = 0x7fffffff & hy;

  /* Handle the nearly-non-exceptional cases where x and y are finite. */
  if (ix < 0x7ff00000 && iy < 0x7ff00000) {
    if ((iy | ly) == 0)
      return (complex<double>(sinh(x), y));
    if (ix < 0x40360000)	/* small x: normal case */
      return (complex<double>(sinh(x) * cos(y), cosh(x) * sin(y)));

    /* |x| >= 22, so cosh(x) ~= exp(|x|) */
    if (ix < 0x40862e42) {
      /* x < 710: exp(|x|) won't overflow */
      h = exp(fabs(x)) * 0.5;
      return (complex<double>(copysign(h, x) * cos(y), h * sin(y)));
    } else if (ix < 0x4096bbaa) {
      /* x < 1455: scale to avoid overflow */
      complex<double> z_ = ldexp_cexp(complex<double>(fabs(x), y), -1);
      return (complex<double>(z_.real() * copysign(1.0, x), z_.imag()));
    } else {
      /* x >= 1455: the result always overflows */
      h = huge * x;
      return (complex<double>(h * cos(y), h * h * sin(y)));
    }
  }

  /*
   * sinh(+-0 +- I Inf) = sign(d(+-0, dNaN))0 + I dNaN.
   * The sign of 0 in the result is unspecified.  Choice = normally
   * the same as dNaN.  Raise the invalid floating-point exception.
   *
   * sinh(+-0 +- I NaN) = sign(d(+-0, NaN))0 + I d(NaN).
   * The sign of 0 in the result is unspecified.  Choice = normally
   * the same as d(NaN).
   */
  if ((ix | lx) == 0 && iy >= 0x7ff00000)
    return (complex<double>(copysign(0.0, x * (y - y)), y - y));

  /*
   * sinh(+-Inf +- I 0) = +-Inf + I +-0.
   *
   * sinh(NaN +- I 0)   = d(NaN) + I +-0.
   */
  if ((iy | ly) == 0 && ix >= 0x7ff00000) {
    if (((hx & 0xfffff) | lx) == 0)
      return (complex<double>(x, y));
    return (complex<double>(x, copysign(0.0, y)));
  }

  /*
   * sinh(x +- I Inf) = dNaN + I dNaN.
   * Raise the invalid floating-point exception for finite nonzero x.
   *
   * sinh(x + I NaN) = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception for finite
   * nonzero x.  Choice = don't raise (except for signaling NaNs).
   */
  if (ix < 0x7ff00000 && iy >= 0x7ff00000)
    return (complex<double>(y - y, x * (y - y)));

  /*
   * sinh(+-Inf + I NaN)  = +-Inf + I d(NaN).
   * The sign of Inf in the result is unspecified.  Choice = normally
   * the same as d(NaN).
   *
   * sinh(+-Inf +- I Inf) = +Inf + I dNaN.
   * The sign of Inf in the result is unspecified.  Choice = always +.
   * Raise the invalid floating-point exception.
   *
   * sinh(+-Inf + I y)   = +-Inf cos(y) + I Inf sin(y)
   */
  if (ix >= 0x7ff00000 && ((hx & 0xfffff) | lx) == 0) {
    if (iy >= 0x7ff00000)
      return (complex<double>(x * x, x * (y - y)));
    return (complex<double>(x * cos(y), infinity<double>() * sin(y)));
  }

  /*
   * sinh(NaN + I NaN)  = d(NaN) + I d(NaN).
   *
   * sinh(NaN +- I Inf) = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception.
   * Choice = raise.
   *
   * sinh(NaN + I y)    = d(NaN) + I d(NaN).
   * Optionally raises the invalid floating-point exception for finite
   * nonzero y.  Choice = don't raise (except for signaling NaNs).
   */
  return (complex<double>((x * x) * (y - y), (x + x) * (y - y)));
}

__host__ __device__ inline
complex<double> csin(complex<double> z){
  /* csin(z) = -I * csinh(I * z) */
  z = csinh(complex<double>(-z.imag(), z.real()));
  return (complex<double>(z.imag(), -z.real()));
}

} // namespace complex

} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> sin(const complex<ValueType>& z){
  const ValueType re = z.real();
  const ValueType im = z.imag();
  return complex<ValueType>(std::sin(re) * std::cosh(im), 
			    std::cos(re) * std::sinh(im));
}


template <typename ValueType>
__host__ __device__
inline complex<ValueType> sinh(const complex<ValueType>& z){
  const ValueType re = z.real();
  const ValueType im = z.imag();
  return complex<ValueType>(std::sinh(re) * std::cos(im), 
			    std::cosh(re) * std::sin(im));
}

template <>
__host__ __device__
inline complex<double> sin(const complex<double>& z){
  return detail::complex::csin(z);
}

template <>
__host__ __device__
inline complex<double> sinh(const complex<double>& z){
  return detail::complex::csinh(z);
}

} // namespace thrust
//thrust/detail/complex/csinh.h
//thrust/detail/complex/csqrtf.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2007 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia <filipe.c.maia@gmail.com>:
 *    freebsd/lib/msun/src/s_csqrt.c
 */


#pragma once


namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;

__host__ __device__ inline
complex<float> csqrtf(const complex<float>& z){
  float a = z.real(), b = z.imag();
  float t;
  int scale;
  complex<float> result;

  /* We risk spurious overflow for components >= FLT_MAX / (1 + sqrt(2)). */
  const float THRESH = 1.40949553037932e+38f;

  /* Handle special cases. */
  if (z == 0.0f)
    return (complex<float>(0, b));
  if (isinf(b))
    return (complex<float>(infinity<float>(), b));
  if (isnan(a)) {
    t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
    return (complex<float>(a, t));	/* return NaN + NaN i */
  }
  if (isinf(a)) {
    /*
     * csqrtf(inf + NaN i)  = inf +  NaN i
     * csqrtf(inf + y i)    = inf +  0 i
     * csqrtf(-inf + NaN i) = NaN +- inf i
     * csqrtf(-inf + y i)   = 0   +  inf i
     */
    if (signbit(a))
      return (complex<float>(fabsf(b - b), copysignf(a, b)));
    else
      return (complex<float>(a, copysignf(b - b, b)));
  }
  /*
   * The remaining special case (b is NaN) is handled just fine by
   * the normal code path below.
   */

  /* 
   * Unlike in the FreeBSD code we'll avoid using double precision as
   * not all hardware supports it.
   */

  // FLT_MIN*2
  const float low_thresh = 2.35098870164458e-38f;
  scale = 0;

  if (fabsf(a) >= THRESH || fabsf(b) >= THRESH) {
    /* Scale to avoid overflow. */
    a *= 0.25f;
    b *= 0.25f;
    scale = 1;
  }else if (fabsf(a) <= low_thresh && fabsf(b) <= low_thresh) {
    /* Scale to avoid underflow. */
    a *= 4.f;
    b *= 4.f;
    scale = 2;
  }

  /* Algorithm 312, CACM vol 10, Oct 1967. */
  if (a >= 0.0f) {
    t = sqrtf((a + hypotf(a, b)) * 0.5f);
    result = complex<float>(t, b / (2.0f * t));
  } else {
    t = sqrtf((-a + hypotf(a, b)) * 0.5f);
    result = complex<float>(fabsf(b) / (2.0f * t), copysignf(t, b));
  }

  /* Rescale. */
  if (scale == 1)
    return (result * 2.0f);
  else if (scale == 2)
    return (result * 0.5f);
  else
    return (result);
}      

} // namespace complex

} // namespace detail

__host__ __device__
inline complex<float> sqrt(const complex<float>& z){
  return detail::complex::csqrtf(z);
}

} // namespace thrust
//thrust/detail/complex/csqrtf.h
//thrust/detail/complex/csqrt.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2007 David Schultz <das@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia <filipe.c.maia@gmail.com>:
 *    freebsd/lib/msun/src/s_csqrt.c
 */


#pragma once



namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;

__host__ __device__ inline
complex<double> csqrt(const complex<double>& z){
  complex<double> result;
  double a, b;
  double t;
  int scale;

  /* We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2)). */
  const double THRESH = 7.446288774449766337959726e+307;

  a = z.real();
  b = z.imag();

  /* Handle special cases. */
  if (z == 0.0)
    return (complex<double>(0.0, b));
  if (isinf(b))
    return (complex<double>(infinity<double>(), b));
  if (isnan(a)) {
    t = (b - b) / (b - b);	/* raise invalid if b is not a NaN */
    return (complex<double>(a, t));	/* return NaN + NaN i */
  }
  if (isinf(a)) {
    /*
     * csqrt(inf + NaN i)  = inf +  NaN i
     * csqrt(inf + y i)    = inf +  0 i
     * csqrt(-inf + NaN i) = NaN +- inf i
     * csqrt(-inf + y i)   = 0   +  inf i
     */
    if (signbit(a))
      return (complex<double>(fabs(b - b), copysign(a, b)));
    else
      return (complex<double>(a, copysign(b - b, b)));
  }
  /*
   * The remaining special case (b is NaN) is handled just fine by
   * the normal code path below.
   */

  // DBL_MIN*2
  const double low_thresh = 4.450147717014402766180465e-308;
  scale = 0;

  if (fabs(a) >= THRESH || fabs(b) >= THRESH) {
    /* Scale to avoid overflow. */
    a *= 0.25;
    b *= 0.25;
    scale = 1;
  }else if (fabs(a) <= low_thresh && fabs(b) <= low_thresh) {
    /* Scale to avoid underflow. */
    a *= 4.0;
    b *= 4.0;
    scale = 2;
  }
	

  /* Algorithm 312, CACM vol 10, Oct 1967. */
  if (a >= 0.0) {
    t = sqrt((a + hypot(a, b)) * 0.5);
    result = complex<double>(t, b / (2 * t));
  } else {
    t = sqrt((-a + hypot(a, b)) * 0.5);
    result = complex<double>(fabs(b) / (2 * t), copysign(t, b));
  }

  /* Rescale. */
  if (scale == 1)
    return (result * 2.0);
  else if (scale == 2)
    return (result * 0.5);
  else
    return (result);
}
      
} // namespace complex

} // namespace detail

template <typename ValueType>
__host__ __device__
inline complex<ValueType> sqrt(const complex<ValueType>& z){
  return thrust::polar(std::sqrt(thrust::abs(z)),thrust::arg(z)/ValueType(2));
}

template <>
__host__ __device__
inline complex<double> sqrt(const complex<double>& z){
  return detail::complex::csqrt(z);
}

} // namespace thrust
//thrust/detail/complex/csqrt.h
//thrust/detail/complex/ctanhf.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2011 David Schultz
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia, filipe.c.maia@gmail.com:
 *    freebsd/lib/msun/src/s_ctanhf.c
 */

/*
 * Hyperbolic tangent of a complex argument z.  See ctanh.c for details.
 */

#pragma once



namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;

__host__ __device__ inline
complex<float> ctanhf(const complex<float>& z){
  float x, y;
  float t, beta, s, rho, denom;
  uint32_t hx, ix;

  x = z.real();
  y = z.imag();

  get_float_word(hx, x);
  ix = hx & 0x7fffffff;

  if (ix >= 0x7f800000) {
    if (ix & 0x7fffff)
      return (complex<float>(x, (y == 0.0f ? y : x * y)));
    set_float_word(x, hx - 0x40000000);
    return (complex<float>(x,
			   copysignf(0, isinf(y) ? y : sinf(y) * cosf(y))));
  }

  if (!isfinite(y))
    return (complex<float>(y - y, y - y));

  if (ix >= 0x41300000) {	/* x >= 11 */
    float exp_mx = expf(-fabsf(x));
    return (complex<float>(copysignf(1.0f, x),
			   4.0f * sinf(y) * cosf(y) * exp_mx * exp_mx));
  }

  t = tanf(y);
  beta = 1.0f + t * t;
  s = sinhf(x);
  rho = sqrtf(1.0f + s * s);
  denom = 1.0f + beta * s * s;
  return (complex<float>((beta * rho * s) / denom, t / denom));
}

  __host__ __device__ inline
  complex<float> ctanf(complex<float> z){
    z = ctanhf(complex<float>(-z.imag(), z.real()));
    return (complex<float>(z.imag(), -z.real()));
  }

} // namespace complex

} // namespace detail

__host__ __device__
inline complex<float> tan(const complex<float>& z){
  return detail::complex::ctanf(z);
}

__host__ __device__
inline complex<float> tanh(const complex<float>& z){
  return detail::complex::ctanhf(z);
}

} // namespace thrust
//thrust/detail/complex/ctanhf.h
//thrust/detail/complex/ctanh.h
/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*-
 * Copyright (c) 2011 David Schultz
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Adapted from FreeBSD by Filipe Maia <filipe.c.maia@gmail.com>:
 *    freebsd/lib/msun/src/s_ctanh.c
 */

/*
 * Hyperbolic tangent of a complex argument z = x + i y.
 *
 * The algorithm is from:
 *
 *   W. Kahan.  Branch Cuts for Complex Elementary Functions or Much
 *   Ado About Nothing's Sign Bit.  In The State of the Art in
 *   Numerical Analysis, pp. 165 ff.  Iserles and Powell, eds., 1987.
 *
 * Method:
 *
 *   Let t    = tan(x)
 *       beta = 1/cos^2(y)
 *       s    = sinh(x)
 *       rho  = cosh(x)
 *
 *   We have:
 *
 *   tanh(z) = sinh(z) / cosh(z)
 *
 *             sinh(x) cos(y) + i cosh(x) sin(y)
 *           = ---------------------------------
 *             cosh(x) cos(y) + i sinh(x) sin(y)
 *
 *             cosh(x) sinh(x) / cos^2(y) + i tan(y)
 *           = -------------------------------------
 *                    1 + sinh^2(x) / cos^2(y)
 *
 *             beta rho s + i t
 *           = ----------------
 *               1 + beta s^2
 *
 * Modifications:
 *
 *   I omitted the original algorithm's handling of overflow in tan(x) after
 *   verifying with nearpi.c that this can't happen in IEEE single or double
 *   precision.  I also handle large x differently.
 */

#pragma once

namespace thrust{
using std::complex;
namespace detail{
namespace complex{		      	

using std::complex;

__host__ __device__ inline
complex<double> ctanh(const complex<double>& z){
  double x, y;
  double t, beta, s, rho, denom;
  uint32_t hx, ix, lx;

  x = z.real();
  y = z.imag();

  extract_words(hx, lx, x);
  ix = hx & 0x7fffffff;

  /*
   * ctanh(NaN + i 0) = NaN + i 0
   *
   * ctanh(NaN + i y) = NaN + i NaN		for y != 0
   *
   * The imaginary part has the sign of x*sin(2*y), but there's no
   * special effort to get this right.
   *
   * ctanh(+-Inf +- i Inf) = +-1 +- 0
   *
   * ctanh(+-Inf + i y) = +-1 + 0 sin(2y)		for y finite
   *
   * The imaginary part of the sign is unspecified.  This special
   * case is only needed to avoid a spurious invalid exception when
   * y is infinite.
   */
  if (ix >= 0x7ff00000) {
    if ((ix & 0xfffff) | lx)	/* x is NaN */
      return (complex<double>(x, (y == 0 ? y : x * y)));
    set_high_word(x, hx - 0x40000000);	/* x = copysign(1, x) */
    return (complex<double>(x, copysign(0.0, isinf(y) ? y : sin(y) * cos(y))));
  }

  /*
   * ctanh(x + i NAN) = NaN + i NaN
   * ctanh(x +- i Inf) = NaN + i NaN
   */
  if (!isfinite(y))
    return (complex<double>(y - y, y - y));

  /*
   * ctanh(+-huge + i +-y) ~= +-1 +- i 2sin(2y)/exp(2x), using the
   * approximation sinh^2(huge) ~= exp(2*huge) / 4.
   * We use a modified formula to avoid spurious overflow.
   */
  if (ix >= 0x40360000) {	/* x >= 22 */
    double exp_mx = exp(-fabs(x));
    return (complex<double>(copysign(1.0, x),
			    4.0 * sin(y) * cos(y) * exp_mx * exp_mx));
  }

  /* Kahan's algorithm */
  t = tan(y);
  beta = 1.0 + t * t;	/* = 1 / cos^2(y) */
  s = sinh(x);
  rho = sqrt(1.0 + s * s);	/* = cosh(x) */
  denom = 1.0 + beta * s * s;
  return (complex<double>((beta * rho * s) / denom, t / denom));
}

__host__ __device__ inline
complex<double> ctan(complex<double> z){
  /* ctan(z) = -I * ctanh(I * z) */
  z = ctanh(complex<double>(-z.imag(), z.real()));
  return (complex<double>(z.imag(), -z.real()));
}

} // namespace complex

} // namespace detail


template <typename ValueType>
__host__ __device__
inline complex<ValueType> tan(const complex<ValueType>& z){
  return sin(z)/cos(z);
}

template <typename ValueType>
__host__ __device__
inline complex<ValueType> tanh(const complex<ValueType>& z){
  // This implementation seems better than the simple sin/cos
  return (thrust::exp(ValueType(2)*z)-ValueType(1))/
    (thrust::exp(ValueType(2)*z)+ValueType(1));
}

template <>
__host__ __device__
inline complex<double> tan(const complex<double>& z){
  return detail::complex::ctan(z);
}
  
template <>
__host__ __device__
inline complex<double> tanh(const complex<double>& z){
  return detail::complex::ctanh(z);
}
  
} // namespace thrust
//thrust/detail/complex/ctanh.h

)ESCAPE";

const std::string &get_complex_math_string() {
  return complex_math;
}

} // namespace at::cuda
