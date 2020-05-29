#pragma once

#include <complex>
#include <iostream>

#include <c10/macros/Macros.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif

namespace c10 {

// c10::complex is an implementation of complex numbers that aims
// to work on all devices supported by PyTorch
//
// Most of the APIs duplicates std::complex
// Reference: https://en.cppreference.com/w/cpp/numeric/complex
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
// - std::complex defines converting constructor between float/double/long double,
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
//     with specialization `complex& operator=(T x)` for float/double/long double
//     Since we only support float and double, on will use `complex& operator=(T x)`
// - Copy assignment operator and converting assignment operator
//   - There is no specialization of converting assignment operators, which type is
//     convertible is soly depend on whether the scalar type is convertable
//
// In addition to the standard assignment, we also provide assignment operators with std and thrust
//
//
// [Casting operators]
//
// std::complex does not have casting operators. We define casting operators casting to std::complex and thrust::complex
//
//
// [Operator ""]
//
// std::complex has custom literals `i`, `if` and `il` defined in namespace `std::literals::complex_literals`.
// We define our own custom literals in the namespace `c10::complex_literals`. Our custom literals does not
// follow the same behavior as in std::complex, instead, we define _if, _id to construct float/double
// complex literals.
//
//
// [real() and imag()]
//
// In C++20, there are two overload of these functions, one it to return the real/imag, another is to set real/imag,
// they are both constexpr. We follow this design.
//
//
// [Operator +=,-=,*=,/=]
//
// Since C++20, these operators become constexpr. In our implementation, they are also constexpr.
//
// There are two types of such operators: operating with a real number, or operating with another complex number.
// For the operating with a real number, the generic template form has argument type `const T &`, while the overload
// for float/double/long double has `T`. We will follow the same type as float/double/long double in std.
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
// [NOTE: Complex Operator Unification]
// Operators currently use a mix of std::complex, thrust::complex, and c10::complex internally.
// The end state is that all operators will use c10::complex internally.  Until then, there may
// be some hacks to support all variants.
//
//
//
// TODO(@zasdfgbnm): c10::complex<c10::Half> is not currently supported, because:
//  - lots of members and functions of c10::Half are not constexpr
//  - thrust::complex only support float and double

template<typename T>
struct complex;

template<typename T>
struct alignas(sizeof(T) * 2) complex_common {
  using value_type = T;

  T storage[2] = {T(), T()};

  constexpr complex_common() = default;
  constexpr complex_common(const T& re, const T& im = T()): storage{re, im} {}
  template<typename U>
  explicit constexpr complex_common(const std::complex<U> &other): complex_common(other.real(), other.imag()) {}
#if defined(__CUDACC__) || defined(__HIPCC__)
  template<typename U>
  explicit C10_HOST_DEVICE complex_common(const thrust::complex<U> &other): complex_common(other.real(), other.imag()) {}
#endif

  constexpr complex<T> &operator =(T re) {
    storage[0] = re;
    storage[1] = 0;
    return static_cast<complex<T> &>(*this);
  }

  constexpr complex<T> &operator +=(T re) {
    storage[0] += re;
    return static_cast<complex<T> &>(*this);
  }

  constexpr complex<T> &operator -=(T re) {
    storage[0] -= re;
    return static_cast<complex<T> &>(*this);
  }

  constexpr complex<T> &operator *=(T re) {
    storage[0] *= re;
    storage[1] *= re;
    return static_cast<complex<T> &>(*this);
  }

  constexpr complex<T> &operator /=(T re) {
    storage[0] /= re;
    storage[1] /= re;
    return static_cast<complex<T> &>(*this);
  }

  template<typename U>
  constexpr complex<T> &operator =(const complex<U> &rhs) {
    storage[0] = rhs.real();
    storage[1] = rhs.imag();
    return static_cast<complex<T> &>(*this);
  }

  template<typename U>
  constexpr complex<T> &operator +=(const complex<U> &rhs) {
    storage[0] += rhs.real();
    storage[1] += rhs.imag();
    return static_cast<complex<T> &>(*this);
  }

  template<typename U>
  constexpr complex<T> &operator -=(const complex<U> &rhs) {
    storage[0] -= rhs.real();
    storage[1] -= rhs.imag();
    return static_cast<complex<T> &>(*this);
  }

  template<typename U>
  constexpr complex<T> &operator *=(const complex<U> &rhs) {
    // (a + bi) * (c + di) = (a*c - b*d) + (a * d + b * c) i
    T a = storage[0];
    T b = storage[1];
    U c = rhs.real();
    U d = rhs.imag();
    storage[0] = a * c - b * d;
    storage[1] = a * d + b * c;
    return static_cast<complex<T> &>(*this);
  }

  template<typename U>
  constexpr complex<T> &operator /=(const complex<U> &rhs) {
    // (a + bi) / (c + di) = (ac + bd)/(c^2 + d^2) + (bc - ad)/(c^2 + d^2) i
    T a = storage[0];
    T b = storage[1];
    U c = rhs.real();
    U d = rhs.imag();
    auto denominator = c * c + d * d;
    storage[0] = (a * c + b * d) / denominator;
    storage[1] = (b * c - a * d) / denominator;
    return static_cast<complex<T> &>(*this);
  }

  template<typename U>
  constexpr complex<T> &operator =(const std::complex<U> &rhs) {
    storage[0] = rhs.real();
    storage[1] = rhs.imag();
    return static_cast<complex<T> &>(*this);
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template<typename U>
  C10_HOST_DEVICE complex<T> &operator =(const thrust::complex<U> &rhs) {
    storage[0] = rhs.real();
    storage[1] = rhs.imag();
    return static_cast<complex<T> &>(*this);
  }
#endif

  template<typename U>
  explicit constexpr operator std::complex<U>() const {
    return std::complex<U>(std::complex<T>(real(), imag()));
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  template<typename U>
  C10_HOST_DEVICE explicit operator thrust::complex<U>() const {
    return static_cast<thrust::complex<U>>(thrust::complex<T>(real(), imag()));
  }
#endif

  constexpr T real() const {
    return storage[0];
  }
  constexpr void real(T value) {
    storage[0] = value;
  }
  constexpr T imag() const {
    return storage[1];
  }
  constexpr void imag(T value) {
    storage[1] = value;
  }
};

// In principle, `using complex_common<??>::complex_common;` should be enough
// to for default constructor, but multiple compilers are having multiple bugs on
// this at different contexts
#define COMPILER_BUGS constexpr complex() = default;

template<>
struct alignas(2*sizeof(float)) complex<float>: public complex_common<float> {
  using complex_common<float>::complex_common;
  COMPILER_BUGS
  explicit constexpr complex(const complex<double> &other);
  using complex_common<float>::operator=;
};

template<>
struct alignas(2*sizeof(double)) complex<double>: public complex_common<double> {
  using complex_common<double>::complex_common;
  COMPILER_BUGS
  constexpr complex(const complex<float> &other);
  using complex_common<double>::operator=;
};

#undef COMPILER_BUGS

constexpr complex<float>::complex(const complex<double> &other): complex_common(other.real(), other.imag()) {}
constexpr complex<double>::complex(const complex<float> &other): complex_common(other.real(), other.imag()) {}

namespace complex_literals {

constexpr complex<float> operator"" _if(long double imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator"" _id(long double imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

constexpr complex<float> operator"" _if(unsigned long long imag) {
  return complex<float>(0.0f, static_cast<float>(imag));
}

constexpr complex<double> operator"" _id(unsigned long long imag) {
  return complex<double>(0.0, static_cast<double>(imag));
}

} // namespace complex_literals


template<typename T>
constexpr c10::complex<T> operator+(const c10::complex<T>& val) {
  return val;
}

template<typename T>
constexpr c10::complex<T> operator-(const c10::complex<T>& val) {
  return c10::complex<T>(-val.real(), -val.imag());
}

template<typename T>
constexpr c10::complex<T> operator+(const c10::complex<T>& lhs, const c10::complex<T>& rhs) {
  c10::complex<T> result = lhs;
  return result += rhs;
}

template<typename T>
constexpr c10::complex<T> operator+(const c10::complex<T>& lhs, const T& rhs) {
  c10::complex<T> result = lhs;
  return result += rhs;
}

template<typename T>
constexpr c10::complex<T> operator+(const T& lhs, const c10::complex<T>& rhs) {
  return c10::complex<T>(lhs + rhs.real(), rhs.imag());
}

template<typename T>
constexpr c10::complex<T> operator-(const c10::complex<T>& lhs, const c10::complex<T>& rhs) {
  c10::complex<T> result = lhs;
  return result -= rhs;
}

template<typename T>
constexpr c10::complex<T> operator-(const c10::complex<T>& lhs, const T& rhs) {
  c10::complex<T> result = lhs;
  return result -= rhs;
}

template<typename T>
constexpr c10::complex<T> operator-(const T& lhs, const c10::complex<T>& rhs) {
  c10::complex<T> result = -rhs;
  return result += lhs;
}

template<typename T>
constexpr c10::complex<T> operator*(const c10::complex<T>& lhs, const c10::complex<T>& rhs) {
  c10::complex<T> result = lhs;
  return result *= rhs;
}

template<typename T>
constexpr c10::complex<T> operator*(const c10::complex<T>& lhs, const T& rhs) {
  c10::complex<T> result = lhs;
  return result *= rhs;
}

template<typename T>
constexpr c10::complex<T> operator*(const T& lhs, const c10::complex<T>& rhs) {
  c10::complex<T> result = rhs;
  return result *= lhs;
}

template<typename T>
constexpr c10::complex<T> operator/(const c10::complex<T>& lhs, const c10::complex<T>& rhs) {
  c10::complex<T> result = lhs;
  return result /= rhs;
}

template<typename T>
constexpr c10::complex<T> operator/(const c10::complex<T>& lhs, const T& rhs) {
  c10::complex<T> result = lhs;
  return result /= rhs;
}

template<typename T>
constexpr c10::complex<T> operator/(const T& lhs, const c10::complex<T>& rhs) {
  c10::complex<T> result(lhs, T());
  return result /= rhs;
}


// Define operators between integral scalars and c10::complex. std::complex does not support this when T is a
// floating-point number. This is useful because it saves a lot of "static_cast" when operate a complex and an integer.
// This makes the code both less verbose and potentially more efficient.
#define COMPLEX_INTEGER_OP_TEMPLATE_CONDITION \
  typename std::enable_if_t<std::is_floating_point<fT>::value && std::is_integral<iT>::value, int> = 0

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator+(const c10::complex<fT>& a, const iT& b) {
  return a + static_cast<fT>(b);
}

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator+(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) + b;
}

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator-(const c10::complex<fT>& a, const iT& b) {
  return a - static_cast<fT>(b);
}

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator-(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) - b;
}

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator*(const c10::complex<fT>& a, const iT& b) {
  return a * static_cast<fT>(b);
}

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator*(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) * b;
}

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator/(const c10::complex<fT>& a, const iT& b) {
  return a / static_cast<fT>(b);
}

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
constexpr c10::complex<fT> operator/(const iT& a, const c10::complex<fT>& b) {
  return static_cast<fT>(a) / b;
}

#undef COMPLEX_INTEGER_OP_TEMPLATE_CONDITION


template<typename T>
constexpr bool operator==(const c10::complex<T>& lhs, const c10::complex<T>& rhs) {
  return (lhs.real() == rhs.real()) && (lhs.imag() == rhs.imag());
}

template<typename T>
constexpr bool operator==(const c10::complex<T>& lhs, const T& rhs) {
  return (lhs.real() == rhs) && (lhs.imag() == T());
}

template<typename T>
constexpr bool operator==(const T& lhs, const c10::complex<T>& rhs) {
  return (lhs == rhs.real()) && (T() == rhs.imag());
}

template<typename T>
constexpr bool operator!=(const c10::complex<T>& lhs, const c10::complex<T>& rhs) {
  return !(lhs == rhs);
}

template<typename T>
constexpr bool operator!=(const c10::complex<T>& lhs, const T& rhs) {
  return !(lhs == rhs);
}

template<typename T>
constexpr bool operator!=(const T& lhs, const c10::complex<T>& rhs) {
  return !(lhs == rhs);
}

template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const c10::complex<T>& x) {
  return (os << static_cast<std::complex<T>>(x));
}

template <typename T, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>& operator>>(std::basic_istream<CharT, Traits>& is, c10::complex<T>& x) {
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

template<typename T>
constexpr T real(const c10::complex<T>& z) {
  return z.real();
}

template<typename T>
constexpr T imag(const c10::complex<T>& z) {
  return z.imag();
}

#if defined(CUDA_VERSION) && (CUDA_VERSION < 10000)
#define CUDA92_BUG(x) thrust::complex<T>(x.real(), x.imag())
#else
#define CUDA92_BUG(x) x
#endif

template<typename T>
C10_HOST_DEVICE T abs(const c10::complex<T>& z) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return thrust::abs(static_cast<thrust::complex<T>>(CUDA92_BUG(z)));
#else
  return std::abs(static_cast<std::complex<T>>(z));
#endif
}

#undef CUDA92_BUG

#ifdef __HIP_PLATFORM_HCC__
#define ROCm_Bug(x)
#else
#define ROCm_Bug(x) x
#endif

template<typename T>
C10_HOST_DEVICE T arg(const c10::complex<T>& z) {
  return ROCm_Bug(std)::atan2(std::imag(z), std::real(z));
}

#undef ROCm_Bug

template<typename T>
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
template<typename T>
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

template<typename T>
C10_HOST_DEVICE c10::complex<T> polar(const T& r, const T& theta = T()) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::polar(r, theta));
#else
  return static_cast<c10::complex<T>>(std::polar(r, theta));
#endif
}

} // namespace c10

// math functions are included in a separate file
#include <c10/util/complex_math.h>
// utilities for complex types
#include <c10/util/complex_utils.h>
