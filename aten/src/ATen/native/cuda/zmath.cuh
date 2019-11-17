#pragma once

#include <c10/util/TypeCast.h>

#include <thrust/complex.h> 

namespace c10 {
  // Specialization of is_complex_t() in c10/util/Half.h for thrust::complex types
  template <typename T>
  struct is_complex_t<thrust::complex<T>> : public std::true_type {};

  #ifdef C10_HOST_DEVICE
  #define ERROR_UNSUPPORTED_CAST assert(false);
  #else
  #define ERROR_UNSUPPORTED_CAST TORCH_CHECK(false, "Unexpected scalar type");
  #endif

  // Specialization of fetch_and_cast in c10/util/TypeCast.h for thrust::complex types
  #define FETCH_AND_CAST_CASE(type, scalartype) case ScalarType::scalartype: return static_cast_with_inter_type<thrust::complex<T>>(*(const type *)ptr);
  template<typename T>
  C10_HOST_DEVICE inline thrust::complex<T> fetch_and_cast(const ScalarType src_type, const void *ptr) {
    switch (src_type) {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(FETCH_AND_CAST_CASE)
      default:
        ERROR_UNSUPPORTED_CAST
    }
    return thrust::complex<T>(0); // just to avoid compiler warning
  }

  // Specialization of cast_and_store in c10/util/TypeCast.h for thrust::complex types
  #define CAST_AND_STORE_CASE(type, scalartype) case ScalarType::scalartype: *(type *)ptr = static_cast_with_inter_type<type>(value); return;
  template<typename T>
  C10_HOST_DEVICE inline void cast_and_store(const ScalarType dest_type, void *ptr, thrust::complex<T> value) {
    switch (dest_type) {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(CAST_AND_STORE_CASE)
      default:;
    }
    ERROR_UNSUPPORTED_CAST
  }

  #undef FETCH_AND_CAST_CASE
  #undef CAST_AND_STORE_CASE
  #undef ERROR_UNSUPPORTED_CAST
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
