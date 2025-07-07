#pragma once

#include <complex>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
#if C10_CLANG_HAS_WARNING("-Wfloat-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wfloat-conversion")
#endif

namespace c10 {

// c10::complex is an implementation of complex numbers that aims
// to work on all devices supported by PyTorch
//
// Most of the APIs duplicates std::complex
// Reference: https://en.cppreference.com/w/cpp/numeric/complex
//
// [NOTE: Complex Operator Unification]
// Operators currently use a mix of std::complex, thrust::complex, and
// c10::complex internally. The end state is that all operators will use
// c10::complex internally.  Until then, there may be some hacks to support all
// variants.
//
//
// [Note on Constructors]
//
// The APIs of constructors are mostly copied from C++ standard:
//   https://en.cppreference.com/w/cpp/numeric/complex/complex
//
// Since C++14, all constructors are constexpr in std::complex
//
// There are three types of constructors:
// - initializing from real and imag:
//     `constexpr complex( const T& re = T(), const T& im = T() );`
// - implicitly-declared copy constructor
// - converting constructors
//
// Converting constructors:
// - std::complex defines converting constructor between float/double/long
// double,
//   while we define converting constructor between float/double.
// - For these converting constructors, upcasting is implicit, downcasting is
//   explicit.
// - We also define explicit casting from std::complex/thrust::complex
//   - Note that the conversion from thrust is not constexpr, because
//     thrust does not define them as constexpr ????
//
//
// [Operator =]
//
// The APIs of operator = are mostly copied from C++ standard:
//   https://en.cppreference.com/w/cpp/numeric/complex/operator%3D
//
// Since C++20, all operator= are constexpr. Although we are not building with
// C++20, we also obey this behavior.
//
// There are three types of assign operator:
// - Assign a real value from the same scalar type
//   - In std, this is templated as complex& operator=(const T& x)
//     with specialization `complex& operator=(T x)` for float/double/long
//     double Since we only support float and double, on will use `complex&
//     operator=(T x)`
// - Copy assignment operator and converting assignment operator
//   - There is no specialization of converting assignment operators, which type
//   is
//     convertible is solely dependent on whether the scalar type is convertible
//
// In addition to the standard assignment, we also provide assignment operators
// with std and thrust
//
//
// [Casting operators]
//
// std::complex does not have casting operators. We define casting operators
// casting to std::complex and thrust::complex
//
//
// [Operator ""]
//
// std::complex has custom literals `i`, `if` and `il` defined in namespace
// `std::literals::complex_literals`. We define our own custom literals in the
// namespace `c10::complex_literals`. Our custom literals does not follow the
// same behavior as in std::complex, instead, we define _if, _id to construct
// float/double complex literals.
//
//
// [real() and imag()]
//
// In C++20, there are two overload of these functions, one it to return the
// real/imag, another is to set real/imag, they are both constexpr. We follow
// this design.
//
//
// [Operator +=,-=,*=,/=]
//
// Since C++20, these operators become constexpr. In our implementation, they
// are also constexpr.
//
// There are two types of such operators: operating with a real number, or
// operating with another complex number. For the operating with a real number,
// the generic template form has argument type `const T &`, while the overload
// for float/double/long double has `T`. We will follow the same type as
// float/double/long double in std.
//
// [Unary operator +-]
//
// Since C++20, they are constexpr. We also make them expr
//
// [Binary operators +-*/]
//
// Each operator has three versions (taking + as example):
// - complex + complex
// - complex + real
// - real + complex
//
// [Operator ==, !=]
//
// Each operator has three versions (taking == as example):
// - complex == complex
// - complex == real
// - real == complex
//
// Some of them are removed on C++20, but we decide to keep them
//
// [Operator <<, >>]
//
// These are implemented by casting to std::complex
//
//
//
// TODO(@zasdfgbnm): c10::complex<c10::Half> is not currently supported,
// because:
//  - lots of members and functions of c10::Half are not constexpr
//  - thrust::complex only support float and double

template <typename T>
struct alignas(sizeof(T) * 2) complex {
  using value_type = T;

  T real_ = T(0);
  T imag_ = T(0);

  constexpr complex() = default;
  C10_HOST_DEVICE constexpr complex(const T& re, const T& im = T())
      : real_(re), imag_(im) {}
  template <typename U>
  explicit constexpr complex(const std::complex<U>& other)
      : complex(other.real(), other.imag()) {}
#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  explicit C10_HOST_DEVICE complex(const thrust::complex<U>& other)
      : real_(other.real()), imag_(other.imag()) {}
// NOTE can not be implemented as follow due to ROCm bug:
//   explicit C10_HOST_DEVICE complex(const thrust::complex<U> &other):
//   complex(other.real(), other.imag()) {}
#endif

  // Use SFINAE to specialize casting constructor for c10::complex<float> and
  // c10::complex<double>
  template <typename U = T>
  C10_HOST_DEVICE explicit constexpr complex(
      const std::enable_if_t<std::is_same_v<U, float>, complex<double>>& other)
      : real_(other.real_), imag_(other.imag_) {}
  template <typename U = T>
  C10_HOST_DEVICE constexpr complex(
      const std::enable_if_t<std::is_same_v<U, double>, complex<float>>& other)
      : real_(other.real_), imag_(other.imag_) {}

  constexpr complex<T>& operator=(T re) {
    real_ = re;
    imag_ = 0;
    return *this;
  }

  constexpr complex<T>& operator+=(T re) {
    real_ += re;
    return *this;
  }

  constexpr complex<T>& operator-=(T re) {
    real_ -= re;
    return *this;
  }

  constexpr complex<T>& operator*=(T re) {
    real_ *= re;
    imag_ *= re;
    return *this;
  }

  constexpr complex<T>& operator/=(T re) {
    real_ /= re;
    imag_ /= re;
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator=(const complex<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator+=(const complex<U>& rhs) {
    real_ += rhs.real();
    imag_ += rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator-=(const complex<U>& rhs) {
    real_ -= rhs.real();
    imag_ -= rhs.imag();
    return *this;
  }

  template <typename U>
  constexpr complex<T>& operator*=(const complex<U>& rhs) {
    // (a + bi) * (c + di) = (a*c - b*d) + (a * d + b * c) i
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }

#ifdef __APPLE__
#define FORCE_INLINE_APPLE __attribute__((always_inline))
#else
#define FORCE_INLINE_APPLE
#endif
  template <typename U>
  constexpr FORCE_INLINE_APPLE complex<T>& operator/=(const complex<U>& rhs)
      __ubsan_ignore_float_divide_by_zero__ {
    // (a + bi) / (c + di) = (ac + bd)/(c^2 + d^2) + (bc - ad)/(c^2 + d^2) i
    // the calculation below follows numpy's complex division
    T a = real_;
    T b = imag_;
    U c = rhs.real();
    U d = rhs.imag();

#if defined(__GNUC__) && !defined(__clang__)
    // std::abs is already constexpr by gcc
    auto abs_c = std::abs(c);
    auto abs_d = std::abs(d);
#else
    auto abs_c = c < 0 ? -c : c;
    auto abs_d = d < 0 ? -d : d;
#endif

    if (abs_c >= abs_d) {
      if (abs_c == U(0) && abs_d == U(0)) {
        /* divide by zeros should yield a complex inf or nan */
        real_ = a / abs_c;
        imag_ = b / abs_d;
      } else {
        auto rat = d / c;
        auto scl = U(1.0) / (c + d * rat);
        real_ = (a + b * rat) * scl;
        imag_ = (b - a * rat) * scl;
      }
    } else {
      auto rat = c / d;
      auto scl = U(1.0) / (d + c * rat);
      real_ = (a * rat + b) * scl;
      imag_ = (b * rat - a) * scl;
    }
    return *this;
  }
#undef FORCE_INLINE_APPLE

  template <typename U>
  constexpr complex<T>& operator=(const std::complex<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  C10_HOST_DEVICE complex<T>& operator=(const thrust::complex<U>& rhs) {
    real_ = rhs.real();
    imag_ = rhs.imag();
    return *this;
  }
#endif

  template <typename U>
  explicit constexpr operator std::complex<U>() const {
    return std::complex<U>(std::complex<T>(real(), imag()));
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template <typename U>
  C10_HOST_DEVICE explicit operator thrust::complex<U>() const {
    return static_cast<thrust::complex<U>>(thrust::complex<T>(real(), imag()));
  }
#endif

  // consistent with NumPy behavior
  explicit constexpr operator bool() const {
    return real() || imag();
  }

  C10_HOST_DEVICE constexpr T real() const {
    return real_;
  }
  constexpr void real(T value) {
    real_ = value;
  }
  C10_HOST_DEVICE constexpr T imag() const {
    return imag_;
  }
  constexpr void imag(T value) {
    imag_ = value;
  }
};

namespace complex_literals {

constexpr complex<float> operator""_if(long double imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator""_id(long double imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

constexpr complex<float> operator""_if(unsigned long long imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator""_id(unsigned long long imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

} // namespace complex_literals

template <typename T>
constexpr complex<T> operator+(const complex<T>& val) {
  return val;
}

template <typename T>
constexpr complex<T> operator-(const complex<T>& val) {
  return complex<T>(-val.real(), -val.imag());
}

template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result += rhs;
}

template <typename T>
constexpr complex<T> operator+(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result += rhs;
}

template <typename T>
constexpr complex<T> operator+(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs + rhs.real(), rhs.imag());
}

template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result -= rhs;
}

template <typename T>
constexpr complex<T> operator-(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result -= rhs;
}

template <typename T>
constexpr complex<T> operator-(const T& lhs, const complex<T>& rhs) {
  complex<T> result = -rhs;
  return result += lhs;
}

template <typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result *= rhs;
}

template <typename T>
constexpr complex<T> operator*(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result *= rhs;
}

template <typename T>
constexpr complex<T> operator*(const T& lhs, const complex<T>& rhs) {
  complex<T> result = rhs;
  return result *= lhs;
}

template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
  complex<T> result = lhs;
  return result /= rhs;
}

template <typename T>
constexpr complex<T> operator/(const complex<T>& lhs, const T& rhs) {
  complex<T> result = lhs;
  return result /= rhs;
}

template <typename T>
constexpr complex<T> operator/(const T& lhs, const complex<T>& rhs) {
  complex<T> result(lhs, T());
  return result /= rhs;
}

// Define operators between integral scalars and c10::complex. std::complex does
// not support this when T is a floating-point number. This is useful because it
// saves a lot of "static_cast" when operate a complex and an integer. This
// makes the code both less verbose and potentially more efficient.
#define COMPLEX_INTEGER_OP_TEMPLATE_CONDITION                 \
  typename std::enable_if_t<                                  \
      std::is_floating_point_v<fT> && std::is_integral_v<iT>, \
      int> = 0

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator+(const c10::complex<fT>& a, const iT& b) {
  return a + static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator+(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) + b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator-(const c10::complex<fT>& a, const iT& b) {
  return a - static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator-(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) - b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator*(const c10::complex<fT>& a, const iT& b) {
  return a * static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator*(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) * b;
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator/(const c10::complex<fT>& a, const iT& b) {
  return a / static_cast<fT>(b);
}

template <typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator/(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) / b;
}

#undef COMPLEX_INTEGER_OP_TEMPLATE_CONDITION

template <typename T>
constexpr bool operator==(const complex<T>& lhs, const complex<T>& rhs) {
  return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template <typename T>
constexpr bool operator==(const complex<T>& lhs, const T& rhs) {
  return (lhs.real() == rhs) && (lhs.imag() == T());
}

template <typename T>
constexpr bool operator==(const T& lhs, const complex<T>& rhs) {
  return (lhs == rhs.real()) && (T() == rhs.imag());
}

template <typename T>
constexpr bool operator!=(const complex<T>& lhs, const complex<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const complex<T>& lhs, const T& rhs) {
  return !(lhs == rhs);
}

template <typename T>
constexpr bool operator!=(const T& lhs, const complex<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
    std::basic_ostream<CharT, Traits>& os,
    const complex<T>& x) {
  return (os << static_cast<std::complex<T>>(x));
}

template <typename T, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(
    std::basic_istream<CharT, Traits>& is,
    complex<T>& x) {
  std::complex<T> tmp;
  is >> tmp;
  x = tmp;
  return is;
}

} // namespace c10

// std functions
//
// The implementation of these functions also follow the design of C++20

namespace std {

template <typename T>
constexpr T real(const c10::complex<T>& z) {
  return z.real();
}

template <typename T>
constexpr T imag(const c10::complex<T>& z) {
  return z.imag();
}

template <typename T>
C10_HOST_DEVICE T abs(const c10::complex<T>& z) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return thrust::abs(static_cast<thrust::complex<T>>(z));
#else
  return std::abs(static_cast<std::complex<T>>(z));
#endif
}

#if defined(USE_ROCM)
#define ROCm_Bug(x)
#else
#define ROCm_Bug(x) x
#endif

template <typename T>
C10_HOST_DEVICE T arg(const c10::complex<T>& z) {
  return ROCm_Bug(std)::atan2(std::imag(z), std::real(z));
}

#undef ROCm_Bug

template <typename T>
constexpr T norm(const c10::complex<T>& z) {
  return z.real() * z.real() + z.imag() * z.imag();
}

// For std::conj, there are other versions of it:
//   constexpr std::complex<float> conj( float z );
//   template< class DoubleOrInteger >
//   constexpr std::complex<double> conj( DoubleOrInteger z );
//   constexpr std::complex<long double> conj( long double z );
// These are not implemented
// TODO(@zasdfgbnm): implement them as c10::conj
template <typename T>
constexpr c10::complex<T> conj(const c10::complex<T>& z) {
  return c10::complex<T>(z.real(), -z.imag());
}

// Thrust does not have complex --> complex version of thrust::proj,
// so this function is not implemented at c10 right now.
// TODO(@zasdfgbnm): implement it by ourselves

// There is no c10 version of std::polar, because std::polar always
// returns std::complex. Use c10::polar instead;

} // namespace std

namespace c10 {

template <typename T>
C10_HOST_DEVICE complex<T> polar(const T& r, const T& theta = T()) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<complex<T>>(thrust::polar(r, theta));
#else
  // std::polar() requires r >= 0, so spell out the explicit implementation to
  // avoid a branch.
  return complex<T>(r * std::cos(theta), r * std::sin(theta));
#endif
}

template <>
struct alignas(4) complex<Half> {
  Half real_;
  Half imag_;

  // Constructors
  complex() = default;
  // Half constructor is not constexpr so the following constructor can't
  // be constexpr
  C10_HOST_DEVICE explicit inline complex(const Half& real, const Half& imag)
      : real_(real), imag_(imag) {}
  C10_HOST_DEVICE inline complex(const c10::complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  // Conversion operator
  inline C10_HOST_DEVICE operator c10::complex<float>() const {
    return {real_, imag_};
  }

  constexpr C10_HOST_DEVICE Half real() const {
    return real_;
  }
  constexpr C10_HOST_DEVICE Half imag() const {
    return imag_;
  }

  C10_HOST_DEVICE complex<Half>& operator+=(const complex<Half>& other) {
    real_ = static_cast<float>(real_) + static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) + static_cast<float>(other.imag_);
    return *this;
  }

  C10_HOST_DEVICE complex<Half>& operator-=(const complex<Half>& other) {
    real_ = static_cast<float>(real_) - static_cast<float>(other.real_);
    imag_ = static_cast<float>(imag_) - static_cast<float>(other.imag_);
    return *this;
  }

  C10_HOST_DEVICE complex<Half>& operator*=(const complex<Half>& other) {
    auto a = static_cast<float>(real_);
    auto b = static_cast<float>(imag_);
    auto c = static_cast<float>(other.real());
    auto d = static_cast<float>(other.imag());
    real_ = a * c - b * d;
    imag_ = a * d + b * c;
    return *this;
  }
};

} // namespace c10

C10_CLANG_DIAGNOSTIC_POP()

#define C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// math functions are included in a separate file
#include <c10/util/complex_math.h> // IWYU pragma: keep
// utilities for complex types
#include <c10/util/complex_utils.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
