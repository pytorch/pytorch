#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/macros/Macros.h>


namespace c10 {

// Note [Implicit conversion between signed and unsigned]
// C and C++ have a lovely set of implicit conversion rules, where casting
// signed integral values to unsigned integral values is always valid
// (it basically treats the value as if using modulo arithmetic), however
// converting negative floating point values to unsigned integral types
// is UB! This means that: (double)-1 -> (int64_t)-1 -> (uint8_t)255 is
// guaranteed to look like this, but we have (double)-1 -> (uint8_t)<ANYTHING>
// because it's UB. This also makes UBSan really angry.
//
// I think those rules are stupid and we really shouldn't conform to them.
// The structs below ensure that for all unsigned types we use (currently
// only uint8_t), we will do an intermediate convertion via int64_t,
// to ensure that any negative values are wrapped around correctly.
//
// Note that conversions from doubles to signed integral types that can't
// represent a particular value after truncating the fracitonal part are UB as well,
// but fixing them is not as simple as adding an int64_t intermediate, beacuse the
// int64_t -> <smaller signed type> conversion is UB for those large values anyway.
// I guess in that case we just have to live with that, but it's definitely less
// surprising than the thing above.
//
// For the curious:
//   https://en.cppreference.com/w/cpp/language/implicit_conversion
//   The relevant paragraph is "Floating-integral conversions".

template <typename T>
struct inter_copy_type {
  using type = T;
};

template <>
struct inter_copy_type<uint8_t> {
  using type = int64_t;
};

template <typename T>
using inter_copy_type_t = typename inter_copy_type<T>::type;

template<typename dest_t, typename src_t>
struct needs_real {
  constexpr static bool value = (is_complex_t<src_t>::value && !is_complex_t<dest_t>::value);
};

template<bool, typename src_t>
struct maybe_real {
  C10_HOST_DEVICE static inline src_t apply(src_t src) {
    return src;
  }
};

template<typename src_t>
struct maybe_real<true, src_t> {
  C10_HOST_DEVICE static inline auto apply(src_t src) -> decltype(src.real()) {
    return src.real();
  }
};


template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
  C10_HOST_DEVICE __ubsan_ignore_float_cast_overflow__ static inline dest_t apply(src_t src) {
    constexpr bool real = needs_real<dest_t, src_t>::value;
    return static_cast<dest_t>(
      static_cast<inter_copy_type_t<dest_t>>(maybe_real<real, src_t>::apply(src)));
  }
};

// Dynamic type casting utils:
// - fetch_and_cast
// - cast_and_store
//
// fetch_and_cast fetch a value with dynamic type specified by a ScalarType
// from a void pointer and cast it to a static type.
//
// cast_and_store casts a static typed value into dynamic type specified
// by a ScalarType, and store it into a void pointer.
//
// NOTE:
//
// Dynamic casting allows us to support type promotion without blowing up
// the combination space: For example, without dynamic cast, in order to
// implement `add_` with type promotion, we would need something like
//
// AT_DISPATCH_ALL_TYPES(output.dtype(),
//    AT_DISPATCH_ALL_TYPES(input1.dtype(),
//       AT_DISPATCH_ALL_TYPES(input2.dtype(),
//           [](arg0_t a, arg1_t b) -> out_t { return a + b; }
//       )
//    )
// )
//
// If we support N dtypes, the above code would generate the a+b kernel for
// all the N * N * N different supported types, the compilation time and
// binary size would become horrible.
//
// Dynamic casting might sounds like a bad idea in terms of performance.
// Especially if you ever do it in a loop, you are going to do a billion tests.
// But in practice it is not as bad as it might look:
//
// - on CPU, this is a branch that always has the same outcome, therefore
//   hopefully the branch predictor could do the job pretty well
// - on GPU, these branches will not diverge, so we could still have the same
//   warp executing the same line of code
// - Most kernels, like `add`, are bandwidth bound, adding a few clock cycles to
//   check an integer does not hurt the performance much because the ALUs would
//   wait for load instructions anyway.
//
// For the discussion and benchmark, refer to:
// - https://github.com/pytorch/pytorch/pull/28343
// - https://github.com/pytorch/pytorch/pull/28344
// - https://github.com/pytorch/pytorch/pull/28345
//

#ifdef C10_HOST_DEVICE
#define ERROR_UNSUPPORTED_CAST CUDA_KERNEL_ASSERT(false);
#else
#define ERROR_UNSUPPORTED_CAST TORCH_CHECK(false, "Unexpected scalar type");
#endif

// Fetch a value with dynamic type src_type from ptr, and cast it to static type dest_t.
#define FETCH_AND_CAST_CASE(type, scalartype) case ScalarType::scalartype: return static_cast_with_inter_type<dest_t, type>::apply(*(const type *)ptr);
template<typename dest_t>
C10_HOST_DEVICE inline dest_t fetch_and_cast(const ScalarType src_type, const void *ptr) {
  switch (src_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(FETCH_AND_CAST_CASE)
    default:
      ERROR_UNSUPPORTED_CAST
  }
  return dest_t(0); // just to avoid compiler warning
}

// Cast a value with static type src_t into dynamic dest_type, and store it to ptr.
#define CAST_AND_STORE_CASE(type, scalartype) case ScalarType::scalartype: *(type *)ptr = static_cast_with_inter_type<type, src_t>::apply(value); return;
template<typename src_t>
C10_HOST_DEVICE inline void cast_and_store(const ScalarType dest_type, void *ptr, src_t value) {
  switch (dest_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(CAST_AND_STORE_CASE)
    default:;
  }
  ERROR_UNSUPPORTED_CAST
}

#define DEFINE_UNCASTABLE(T, scalartype_)                                                         \
template<>                                                                                        \
C10_HOST_DEVICE inline T fetch_and_cast<T>(const ScalarType src_type, const void *ptr) {          \
  CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == src_type);                                        \
  return *(const T *)ptr;                                                                         \
}                                                                                                 \
template<>                                                                                        \
C10_HOST_DEVICE inline void cast_and_store<T>(const ScalarType dest_type, void *ptr, T value) {   \
  CUDA_KERNEL_ASSERT(ScalarType::scalartype_ == dest_type);                                       \
  *(T *)ptr = value;                                                                              \
}

AT_FORALL_QINT_TYPES(DEFINE_UNCASTABLE)

#undef FETCH_AND_CAST_CASE
#undef CAST_AND_STORE_CASE
#undef DEFINE_UNCASTABLE
#undef ERROR_UNSUPPORTED_CAST

template <typename To, typename From>
To convert(From f) {
  return static_cast_with_inter_type<To, From>::apply(f);
}

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  // Converting to bool can't overflow so we exclude this case from checking.
  if (!std::is_same<To, bool>::value && overflows<To, From>(f)) {
    std::ostringstream oss;
    oss << "value cannot be converted to type " << name
        << " without overflow: " << f;
    throw std::runtime_error(oss.str());  // rather than domain_error (issue 33562)
  }
  return convert<To, From>(f);
}

}  // namespace c10
