#pragma once

#include <c10/util/TypeCast.h>

#include <thrust/complex.h>

namespace c10 {
  // Specialization of is_complex_t() in c10/util/Half.h for thrust::complex types
  template <typename T>
  struct is_complex_t<thrust::complex<T>> : public std::true_type {};

  template <typename dest_t_value_t, typename src_t_value_t>
  struct static_cast_with_inter_type<thrust::complex<dest_t_value_t>, std::complex<src_t_value_t>> {
    C10_HOST_DEVICE static inline thrust::complex<dest_t_value_t> apply(std::complex<src_t_value_t> src) {
      return thrust::complex<dest_t_value_t>(src.real(), src.imag());
    }
  };

  template <typename dest_t_value_t, typename src_t_value_t>
  struct static_cast_with_inter_type<std::complex<dest_t_value_t>, thrust::complex<src_t_value_t>> {
    C10_HOST_DEVICE static inline std::complex<dest_t_value_t> apply(thrust::complex<src_t_value_t> src) {
      return reinterpret_cast<std::complex<dest_t_value_t>&>(src);
    }
  };
}


namespace at { namespace native {
namespace {

template <typename TYPE>
struct ztype_cuda {
  using value_t = TYPE; // Complex template type
  using thrust_t = TYPE; // Equivalent thrust type
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
