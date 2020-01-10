#pragma once

#include <c10/util/TypeCast.h>

#include <thrust/complex.h>

namespace c10 {
  // Specialization of is_complex_t() in c10/util/Half.h for thrust::complex types
  template <typename T>
  struct is_complex_t<thrust::complex<T>> : public std::true_type {};

  // Specialization of static_cast_with_inter_type in c10/util/TypeCast.h to cast
  // from std::complex<src_value_t> to thrust::complex<dest_value_t>
  template <typename dest_value_t, typename src_value_t>
  struct static_cast_with_inter_type<thrust::complex<dest_value_t>, std::complex<src_value_t>> {
    C10_HOST_DEVICE static inline thrust::complex<dest_value_t> apply(std::complex<src_value_t> src) {
      return thrust::complex<dest_value_t>(src.real(), src.imag());
    }
  };

  // Specialization of static_cast_with_inter_type in c10/util/TypeCast.h to cast
  // from thrust::complex<src_value_t> to std::complex<dest_value_t>.
  // Note Binary compatibility is assumed.
  template <typename dest_value_t, typename src_value_t>
  struct static_cast_with_inter_type<std::complex<dest_value_t>, thrust::complex<src_value_t>> {
    C10_HOST_DEVICE static inline std::complex<dest_value_t> apply(thrust::complex<src_value_t> src) {
      return std::complex<dest_value_t>(src.real(), src.imag());
    }
  };
} //end c10

namespace at { namespace native {
namespace {

template <typename TYPE>
struct ztype_cuda {
  using value_t = TYPE; // Complex value_type
  using thrust_t = TYPE; // Equivalent thrust_type
};

template <>
struct ztype_cuda<std::complex<float>> {
  using value_t = float;
  using thrust_t = thrust::complex<float>;
};

template <>
struct ztype_cuda<std::complex<double>> {
  using value_t = double;
  using thrust_t = thrust::complex<double>;
};

} // end namespace
}} //end at::native
