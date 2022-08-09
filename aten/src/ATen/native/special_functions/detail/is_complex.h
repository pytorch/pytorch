#pragma once

#include <complex>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
struct is_complex : public std::false_type {};

template<typename T1>
struct is_complex<const T1> : public is_complex<T1> {};

template<typename T1>
struct is_complex<volatile T1> : public is_complex<T1> {};

template<typename T1>
struct is_complex<const volatile T1> : public is_complex<T1> {};

template<typename T1>
struct is_complex<std::complex<T1>> : public std::true_type {};
}
}
}
}
